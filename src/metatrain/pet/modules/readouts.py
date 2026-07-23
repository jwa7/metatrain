"""Linear readout ("last layer") modules for PET, with optional atom-type gating.

The readout maps last-layer (head) features to a block's output dimension. It is
always a *linear* map — any nonlinearity lives in the per-block heads that
precede it (see ``head_type`` / ``num_head_layers``). Two flavours of atom-type
gating are provided:

* :class:`LinearReadout` — a single shared linear map (``gated=False``, the
  vanilla readout) or one independent linear map per atom-type group
  (``gated=True``, "one-hot" conditioning). Groups are the central-atom type for
  per-atom(-derived) contributions, or the flat pair index ``Z_I * n + Z_J`` for
  per-atom-pair targets.
* :class:`MoEReadout` — a mixture of linear experts gated by routing weights
  derived from an embedding of the central-atom type. Per-atom targets only.

Both modules share the forward signature ``(features, group_idx) -> predictions``
so they are interchangeable, and both are TorchScript-compatible.
"""

import math
from typing import List

import torch


class LinearReadout(torch.nn.Module):
    """Linear readout, optionally atom-type gated ("one-hot").

    :param in_features: Input (head) feature dimension.
    :param out_features: Output feature dimension for the block.
    :param n_groups: Number of gating groups (ignored when ``gated=False``).
        ``n_species`` for central-atom conditioning, ``n_species**2`` for
        per-atom-pair conditioning.
    :param gated: If ``False``, a single shared linear map is used and
        ``group_idx`` is ignored. If ``True``, an independent weight/bias is
        selected per row via ``group_idx``.
    :param chunk_size: When ``gated=True`` and the gather path is taken, the
        per-row weight gather ``weight[group_idx]`` is done in chunks of at most
        ``chunk_size`` rows (atoms, or NEF-flattened pairs), bounding the
        materialised ``(chunk_size, out, in)`` gather tensor instead of scaling
        with the full row count. This is a memory/performance knob only and does
        not affect results: ``chunk_size >= n_rows`` runs a single gather +
        matmul (identical to no chunking), while a smaller value trades a short
        Python loop for a smaller peak memory footprint. Ignored when
        ``gated=False``.
    :param sorted_min_rows: Minimum number of rows for the grouped-matmul path
        (see :meth:`forward`). Below it, the sort and its device-to-host
        synchronisation cost more than the weight traffic they save.
    :param sorted_min_rows_per_group: Minimum average number of rows per group
        (``n_rows / n_groups``) for the grouped-matmul path. Below it, the
        per-group matmuls are too small to amortise the Python loop over the
        groups. Both thresholds are performance knobs only; the two paths give
        identical results.
    """

    gated: bool
    chunk_size: int
    n_groups: int
    sorted_min_rows: int
    sorted_min_rows_per_group: int

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_groups: int,
        gated: bool,
        chunk_size: int = 1024,
        sorted_min_rows: int = 8192,
        sorted_min_rows_per_group: int = 128,
    ) -> None:
        super().__init__()
        self.gated = gated
        self.chunk_size = chunk_size
        self.n_groups = n_groups
        self.sorted_min_rows = sorted_min_rows
        self.sorted_min_rows_per_group = sorted_min_rows_per_group
        self.in_features = in_features
        self.out_features = out_features

        if gated:
            weight = torch.empty(n_groups, out_features, in_features)
            bias = torch.empty(n_groups, out_features)
        else:
            weight = torch.empty(out_features, in_features)
            bias = torch.empty(out_features)
        # Match torch.nn.Linear's default initialisation. On the gated 3-D weight
        # this has to be done one group at a time: kaiming_uniform_ would read
        # fan_in = out_features * in_features from the full tensor instead of
        # in_features, shrinking the initial scale by ~sqrt(out_features).
        if gated:
            for group in range(n_groups):
                torch.nn.init.kaiming_uniform_(weight[group], a=math.sqrt(5))
        else:
            torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        bound = 1.0 / math.sqrt(in_features) if in_features > 0 else 0.0
        torch.nn.init.uniform_(bias, -bound, bound)
        self.weight = torch.nn.Parameter(weight)
        self.bias = torch.nn.Parameter(bias)

    def forward(
        self, features: torch.Tensor, group_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        :param features: ``(n, in_features)`` (node features, or flattened
            per-atom-pair features), or ``(n, n_neighbours, in_features)`` (edge
            features).
        :param group_idx: Long tensor of shape ``(n,)`` selecting the gating
            group for each row of ``features`` (ignored when ``gated=False``).
        :return: Same leading dims as ``features`` with last dim
            ``out_features``.
        """
        if not self.gated:
            return torch.nn.functional.linear(features, self.weight, self.bias)

        # Gated: apply a per-row (out, in) weight and (out,) bias. Promote 2-D
        # features to 3-D so both the node and edge cases are handled uniformly.
        is_2d = features.dim() == 2
        if is_2d:
            features = features.unsqueeze(1)  # (n, 1, in)

        # Two paths compute the same thing but move very different amounts of
        # weight memory. The gather path copies a weight matrix per row, i.e.
        # n_rows * out * in of traffic; the grouped path sorts the rows and reads
        # each of the n_groups matrices once. Sorting only pays off with enough
        # rows in total (it costs a device-to-host sync) and enough rows per group
        # (each group is one matmul in a Python loop), which is exactly the
        # central-atom conditioning regime; per-atom-pair conditioning has
        # n_species**2 groups and lands on the gather path.
        n_rows = features.shape[0]
        use_sorted = (
            n_rows >= self.sorted_min_rows
            and n_rows >= self.sorted_min_rows_per_group * self.n_groups
        )
        if use_sorted:
            out = self._forward_sorted(features, group_idx)
        else:
            out = self._forward_gathered(features, group_idx)

        if is_2d:
            out = out.squeeze(1)
        return out

    def _forward_gathered(
        self, features: torch.Tensor, group_idx: torch.Tensor
    ) -> torch.Tensor:
        """Gather a weight per row and batch-matmul, in chunks of rows.

        The gather ``weight[group_idx]`` materialises an ``(n_rows, out, in)``
        tensor, which can be large. Processing the rows in chunks of at most
        ``chunk_size`` materialises only ``(chunk_size, out, in)`` at a time;
        ``chunk_size >= n_rows`` runs a single chunk, i.e. the plain batched
        gather + matmul.

        :param features: ``(n_rows, n_columns, in_features)`` features.
        :param group_idx: Long tensor of shape ``(n_rows,)``.
        :return: ``(n_rows, n_columns, out_features)`` outputs.
        """
        n_rows = features.shape[0]
        out = torch.empty(
            n_rows,
            features.shape[1],
            self.out_features,
            dtype=features.dtype,
            device=features.device,
        )
        chunk = self.chunk_size
        n_chunks = (n_rows + chunk - 1) // chunk
        for ci in range(n_chunks):
            start = ci * chunk
            end = min(start + chunk, n_rows)
            idx = group_idx[start:end]
            w = self.weight[idx]  # (chunk, out, in)
            b = self.bias[idx]  # (chunk, out)
            out[start:end] = (
                torch.matmul(features[start:end], w.transpose(-2, -1))
                + b.unsqueeze(1)
            )
        return out

    def _forward_sorted(
        self, features: torch.Tensor, group_idx: torch.Tensor
    ) -> torch.Tensor:
        """Sort the rows by group and run one dense linear per non-empty group.

        Each weight matrix is read once instead of being copied per row, and the
        work runs as a few large matmuls rather than many single-row ones.

        :param features: ``(n_rows, n_columns, in_features)`` features.
        :param group_idx: Long tensor of shape ``(n_rows,)``.
        :return: ``(n_rows, n_columns, out_features)`` outputs, in the original
            row order.
        """
        order = torch.argsort(group_idx)
        counts: List[int] = torch.bincount(group_idx, minlength=self.n_groups).tolist()
        sorted_features = features.index_select(0, order)

        out = torch.empty(
            features.shape[0],
            features.shape[1],
            self.out_features,
            dtype=features.dtype,
            device=features.device,
        )
        start = 0
        for group in range(self.n_groups):
            count = counts[group]
            if count > 0:
                out[start : start + count] = torch.nn.functional.linear(
                    sorted_features[start : start + count],
                    self.weight[group],
                    self.bias[group],
                )
            start += count
        return out.index_select(0, torch.argsort(order))


class MoEReadout(torch.nn.Module):
    """Mixture-of-experts linear readout, routed by an atom-type embedding.

    A pool of ``num_experts`` linear experts, split into ``num_routed_experts``
    routed experts and ``num_experts - num_routed_experts`` shared experts. For
    each atom the router (a small central-atom-type embedding) produces softmax
    scores over the routed experts; the top-``num_topk_experts`` are kept
    (sparse gating) and combined with their routing weights. Shared experts are
    always active with unit weight. Conditioning is on the central-atom type
    only, so this is intended for per-atom (or per-atom-derived) targets.

    All routed experts are evaluated for every atom and then masked by the
    sparse routing weights; this keeps the module free of data-dependent control
    flow (TorchScript-friendly). Zero-weighted experts receive no gradient.

    :param in_features: Input (head) feature dimension.
    :param out_features: Output feature dimension for the block.
    :param n_species: Number of distinct atomic species.
    :param num_experts: Total number of experts N (>= 1).
    :param num_routed_experts: Number of Z-gated (routed) experts I (1 <= I <= N).
    :param num_topk_experts: Routed experts kept per atom via TopK (1 <= K' <= I).
    :param embedding_dim: Latent dimension of the species router embedding.
    """

    num_topk: int

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_species: int,
        num_experts: int,
        num_routed_experts: int,
        num_topk_experts: int,
        embedding_dim: int = 16,
    ) -> None:
        super().__init__()

        num_shared_experts = num_experts - num_routed_experts
        if num_routed_experts < 1:
            raise ValueError(
                f"num_routed_experts must be >= 1, got {num_routed_experts}. "
                "Use atom_type_gating='one-hot' or false instead."
            )
        if num_shared_experts < 0:
            raise ValueError(
                f"num_routed_experts ({num_routed_experts}) exceeds "
                f"num_experts ({num_experts})."
            )
        if num_topk_experts < 1 or num_topk_experts > num_routed_experts:
            raise ValueError(
                f"num_topk_experts = {num_topk_experts}; must satisfy "
                f"1 <= num_topk_experts <= num_routed_experts "
                f"({num_routed_experts})."
            )

        self.num_topk = num_topk_experts

        # Router: species_idx -> (n_atoms, I) softmax scores.
        self.species_embedding = torch.nn.Embedding(n_species, embedding_dim)
        self.routing_matrix = torch.nn.Linear(
            embedding_dim, num_routed_experts, bias=False
        )

        # Experts are plain (ungated) linear readouts.
        self.routed_experts = torch.nn.ModuleList(
            [
                LinearReadout(in_features, out_features, 1, gated=False)
                for _ in range(num_routed_experts)
            ]
        )
        self.shared_experts = torch.nn.ModuleList(
            [
                LinearReadout(in_features, out_features, 1, gated=False)
                for _ in range(num_shared_experts)
            ]
        )

    def forward(
        self, features: torch.Tensor, species_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        :param features: ``(n_atoms, in_features)`` or
            ``(n_atoms, n_neighbours, in_features)``.
        :param species_idx: Long tensor of shape ``(n_atoms,)`` with the
            central-atom species index.
        :return: Same leading dims as ``features`` with last dim ``out_features``.
        """
        # Routing: per-atom sparse gating weights over the routed experts.
        u = torch.nn.functional.silu(self.species_embedding(species_idx))
        scores = torch.softmax(self.routing_matrix(u), dim=-1)  # (n_atoms, I)
        topk_scores, topk_idx = torch.topk(scores, self.num_topk, dim=-1)
        routing_weights = torch.zeros_like(scores).scatter(-1, topk_idx, topk_scores)

        # Routed experts: evaluate all, then combine with sparse weights.
        routed_outs: List[torch.Tensor] = torch.jit.annotate(List[torch.Tensor], [])
        for expert in self.routed_experts:
            routed_outs.append(expert(features, species_idx))
        stacked = torch.stack(routed_outs, dim=1)  # (n_atoms, I, ...)

        if stacked.dim() == 3:
            # Node features: (n_atoms, I, out) -> weights (n_atoms, I, 1)
            output = (routing_weights.unsqueeze(-1) * stacked).sum(1)
        else:
            # Edge features: (n_atoms, I, n_nbr, out) -> weights (n_atoms, I, 1, 1)
            output = (routing_weights.unsqueeze(-1).unsqueeze(-1) * stacked).sum(1)

        # Shared experts: always active, unit weight.
        for expert in self.shared_experts:
            output = output + expert(features, species_idx)

        return output
