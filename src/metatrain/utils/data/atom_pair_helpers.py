from typing import Callable, Dict, List, Optional, Tuple

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import NeighborListOptions, System

from metatrain.utils.data.target_info import TargetInfo


def check_no_atom_pair_targets(
    targets: Dict[str, TargetInfo], architecture_name: str
) -> None:
    """
    Raise a clear error if any of ``targets`` has ``sample_kind == "atom_pair"``.

    This is used by architectures that do not yet support atom-pair targets.

    :param targets: Dict mapping target names to their :py:class:`TargetInfo`, e.g.
        ``dataset_info.targets``.
    :param architecture_name: Name of the calling architecture, used in the error
        message.
    :raises ValueError: If any target has ``sample_kind == "atom_pair"``.
    """
    unsupported = [
        name for name, info in targets.items() if info.sample_kind == "atom_pair"
    ]
    if unsupported:
        raise ValueError(
            f"the '{architecture_name}' architecture does not yet support "
            f"'atom_pair' sample-kind targets: {unsupported}."
        )


def get_pair_sample_labels(
    sample_labels: Labels,
    centers: Optional[torch.Tensor] = None,
    neighbors: Optional[torch.Tensor] = None,
    cell_shifts: Optional[torch.Tensor] = None,
    systems: Optional[List[System]] = None,
    nl_options: Optional[NeighborListOptions] = None,
) -> Labels:
    """
    Create per-pair sample labels from center and neighbor atom indices and cell shifts.

    Each row in the returned Labels corresponds to one directed edge (center → neighbor)
    in the neighbor list, identified in the same way as a standard metatensor neighbor
    list: by system index, center atom index, neighbor atom index, and the integer cell
    shift vector ``(cell_shift_a, cell_shift_b, cell_shift_c)``.

    Either ``centers``, ``neighbors`` and ``cell_shifts`` must all be provided directly
    (already flat and batch-offset, as e.g. produced internally by PET's own
    preprocessing), or ``systems`` and ``nl_options`` must be provided so that they can
    be computed by reading each system's own neighbor list.

    :param sample_labels: Labels for all atoms in the batch, with dimensions
        ``["system", "atom"]``, as returned by :func:`get_per_atom_sample_labels`.
    :param centers: Flat tensor of center atom global indices, shape ``(n_edges,)``.
    :param neighbors: Flat tensor of neighbor atom global indices, shape ``(n_edges,)``.
    :param cell_shifts: Integer cell shift vectors for each edge, shape ``(n_edges,
        3)``.
    :param systems: List of systems in the batch. Only used if ``centers``,
        ``neighbors`` and ``cell_shifts`` are not provided.
    :param nl_options: Options for the neighbor list used to enumerate edges. Only used
        if ``centers``, ``neighbors`` and ``cell_shifts`` are not provided.
    :return: Labels with columns ``["system", "first_atom", "second_atom",
        "cell_shift_a", "cell_shift_b", "cell_shift_c"]``, shape ``(n_edges, 6)``.
    """
    if centers is None or neighbors is None or cell_shifts is None:
        if systems is None or nl_options is None:
            raise ValueError(
                "either `centers`, `neighbors` and `cell_shifts` must all be "
                "provided, or `systems` and `nl_options` must be provided so "
                "they can be computed from the systems' neighbor lists"
            )
        centers, neighbors, cell_shifts = _pair_arrays_from_neighbor_lists(
            systems, nl_options
        )

    sample_values = sample_labels.values  # (n_atoms, 2): [system, atom]
    center_values = sample_values[centers]  # (n_edges, 2): [system, first_atom]
    neighbor_values = sample_values[neighbors]  # (n_edges, 2): [system, second_atom]

    pair_values = torch.cat(
        [
            center_values[:, :1],  # system        (n_edges, 1)
            center_values[:, 1:],  # first_atom    (n_edges, 1)
            neighbor_values[:, 1:],  # second_atom   (n_edges, 1)
            cell_shifts,  # a, b, c       (n_edges, 3)
        ],
        dim=1,
    )

    return Labels(
        names=[
            "system",
            "first_atom",
            "second_atom",
            "cell_shift_a",
            "cell_shift_b",
            "cell_shift_c",
        ],
        values=pair_values,
        assume_unique=True,
    )


def _pair_arrays_from_neighbor_lists(
    systems: List[System],
    nl_options: NeighborListOptions,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reads each system's own neighbor list and returns the flat, batch-offset
    ``centers``, ``neighbors`` and ``cell_shifts`` tensors required by
    :func:`get_pair_sample_labels`.

    :param systems: List of systems in the batch.
    :param nl_options: Options for the neighbor list used to enumerate edges.
    :return: Tuple of ``(centers, neighbors, cell_shifts)``.
    """
    device = systems[0].positions.device
    nl_values_list: List[torch.Tensor] = []
    num_edges: List[int] = []
    node_offsets_list: List[int] = []

    node_counter = 0
    for system in systems:
        assert len(system.known_neighbor_lists()) >= 1, "no neighbor list found"
        neighbor_list = system.get_neighbor_list(nl_options)
        nl_values = neighbor_list.samples.values
        nl_values_list.append(nl_values)

        system_size = len(system)
        node_offsets_list.append(node_counter)
        num_edges.append(nl_values.shape[0])
        node_counter += system_size

    nl_values = torch.cat(nl_values_list)
    centers = nl_values[:, 0]
    neighbors = nl_values[:, 1]
    cell_shifts = nl_values[:, 2:]

    # Compute the offsets
    total_edges = sum(num_edges)
    num_edges_tensor = torch.tensor(num_edges, device=device, dtype=torch.long)
    node_offsets = torch.tensor(node_offsets_list, device=device, dtype=torch.long)
    edge_offsets = torch.repeat_interleave(
        node_offsets, num_edges_tensor, output_size=total_edges
    ).to(dtype=centers.dtype)

    # Offset the centers and neighbors
    centers = centers + edge_offsets
    neighbors = neighbors + edge_offsets

    return centers, neighbors, cell_shifts


def get_single_direction_edges(tmap: TensorMap) -> TensorMap:
    """Takes a TensorMap with edges in both directions and returns a
    TensorMap with only one direction of the edges.

    It keeps only:

    - If atom types are different, the edges where the first atom type
      is smaller than the second atom type.
    - If atom types are the same, the edges where the first atom index
      is smaller than the second atom index.

    This function only works for atomic basis data for now.

    :param tmap: A TensorMap containing edge data in both directions.
    :return: A TensorMap containing edge data in only one direction.
    """
    is_atomic_basis = "first_atom_type" in tmap.keys.names
    if not is_atomic_basis:
        raise ValueError(
            "Getting single direction edges is only supported for "
            "atomic basis data for now."
        )

    new_blocks = []
    new_keys = []
    for key, block in tmap.items():
        # If atom types are different, drop blocks where the first
        # atom type is greater than the second atom type, and keep
        # the block untouched when the first atom type is smaller.
        # (this is the easiest case)
        if key["first_atom_type"] > key["second_atom_type"]:
            continue
        elif key["first_atom_type"] < key["second_atom_type"]:
            new_blocks.append(block)
            new_keys.append(key)
            continue
        else:
            # Otherwise, get the edges where the first atom index is smaller
            # than the second one.
            mask = block.samples["first_atom"] < block.samples["second_atom"]
            new_block = TensorBlock(
                values=block.values[mask],
                samples=Labels(
                    names=block.samples.names,
                    values=block.samples.values[mask],
                ),
                components=block.components,
                properties=Labels(
                    names=block.properties.names, values=block.properties.values
                ),
            )
            new_blocks.append(new_block)

            new_keys.append(key)

    return TensorMap(
        blocks=new_blocks,
        keys=Labels(
            names=tmap.keys.names, values=torch.tensor(new_keys, device=tmap.device)
        ),
    )


def get_bidirectional_edges(tmap: TensorMap) -> TensorMap:
    """Takes a TensorMap with edges in only one direction and returns a
    TensorMap with edges in both directions.

    This function only supports atomic basis data that comes from a coupled
    product for now.

    :param tmap: A TensorMap containing edge data in only one direction.
    :return: A TensorMap containing edge data in both directions.
    """
    is_atomic_basis = "first_atom_type" in tmap.keys.names
    is_coupled = "n_1" in tmap.block(0).properties.names

    if not is_atomic_basis or not is_coupled:
        raise ValueError(
            "Getting multi direction edges is only supported for "
            "atomic basis data coming from a coupled product for now."
        )

    # Get the indices of keys, samples and properties fields so that
    # we can permute them.
    i_first = tmap.block(0).samples.names.index("first_atom")
    i_second = tmap.block(0).samples.names.index("second_atom")
    cell_shift_a = tmap.block(0).samples.names.index("cell_shift_a")
    cell_shift_b = tmap.block(0).samples.names.index("cell_shift_b")
    cell_shift_c = tmap.block(0).samples.names.index("cell_shift_c")
    if is_coupled:
        i_n1 = tmap.block(0).properties.names.index("n_1")
        i_n2 = tmap.block(0).properties.names.index("n_2")
        i_l1 = tmap.block(0).properties.names.index("l_1")
        i_l2 = tmap.block(0).properties.names.index("l_2")
    if is_atomic_basis:
        i_type1 = tmap.keys.names.index("first_atom_type")
        i_type2 = tmap.keys.names.index("second_atom_type")

    new_blocks = []
    new_keys = []
    for key, block in tmap.items():
        if is_atomic_basis:
            # If the block corresponds to edges with different atom types,
            # we keep the block untouched, and we will also create the
            # reverse block.
            if key["first_atom_type"] < key["second_atom_type"]:
                new_blocks.append(block)
                new_keys.append(key.values)
            elif key["first_atom_type"] > key["second_atom_type"]:
                raise ValueError(
                    "Expected the input TensorMap to only contain one direction "
                    "of the edges."
                )

        # Get the reverse connections (i -> j becomes j -> i, and the supercell
        # shift is reversed).
        reverse_samples = block.samples.values.clone()
        reverse_samples[:, [i_first, i_second]] = reverse_samples[
            :, [i_second, i_first]
        ]
        reverse_samples[:, [cell_shift_a, cell_shift_b, cell_shift_c]] *= -1

        # Get the values for the data of the reverse connections.
        reverse_values = block.values
        if is_coupled:
            # If o3_sigma is -1, the reverse block should have its values negated.
            reverse_values = reverse_values * key["o3_sigma"]

        # Get the properties of the reverse block.
        properties = block.properties.values.clone()
        if is_coupled:
            # Swap n_1 with n_2, and l_1 with l_2.
            properties[:, [i_n1, i_n2, i_l1, i_l2]] = properties[
                :, [i_n2, i_n1, i_l2, i_l1]
            ]
        reverse_properties = Labels(names=block.properties.names, values=properties)

        # Now we can construct the final block.
        if is_atomic_basis and key["first_atom_type"] < key["second_atom_type"]:
            # Block with only the reverse connections.
            # This is because we are creating the block with first_atom_type greater
            # than second_atom_type (the opposite one we already have it, see above).
            new_block = TensorBlock(
                values=reverse_values,
                samples=Labels(
                    names=block.samples.names,
                    values=reverse_samples,
                ),
                components=block.components,
                properties=reverse_properties,
            )
            new_key = key.values.clone()
            new_key[[i_type1, i_type2]] = new_key[[i_type2, i_type1]]
        else:
            # Block containing both connections.
            selection = block.properties.select(reverse_properties)
            new_block = TensorBlock(
                values=torch.cat([block.values, reverse_values[..., selection]], dim=0),
                samples=Labels(
                    names=block.samples.names,
                    values=torch.cat([block.samples.values, reverse_samples], dim=0),
                ),
                components=block.components,
                properties=block.properties,
            )
            new_key = key.values

        new_blocks.append(new_block)
        new_keys.append(new_key)

    return TensorMap(
        blocks=new_blocks,
        keys=Labels(names=tmap.keys.names, values=torch.stack(new_keys)),
    )


def get_bidirectional_edges_transform(
    target_info_dict: dict[str, TargetInfo],
    extra_data_info_dict: dict[str, TargetInfo],
) -> tuple[Callable, Callable]:
    """
    Get transform functions to go from single direction edges to bidirectional
    edges and the reverse.

    :param target_info_dict: Dictionary mapping target names to TargetInfo objects.
    :param extra_data_info_dict: Dictionary mapping extra data names to TargetInfo
        objects.

    :return: Two functions: the first one transforms single direction edges to
      bidirectional and the second one transforms bidirectional edges to
      single direction.
    """

    def transform(
        systems: list[System],
        targets: dict[str, TensorMap],
        extra: dict[str, TensorMap],
    ) -> tuple[list[System], dict[str, TensorMap], dict[str, TensorMap]]:
        """
        Transform function that gets the bidirectional edges from the single direction
        ones.

        :param systems: List of systems.
        :param targets: Dictionary containing the targets corresponding to the systems.
        :param extra: Dictionary containing any extra data.
        :return: The systems, targets and extra data with bidirectional data
           for the edges.
        """
        for name, tensor in targets.items():
            if (
                name in target_info_dict
                and target_info_dict[name].sample_kind == "atom_pair"
            ):
                targets[name] = get_bidirectional_edges(tensor)

        for name, tensor in extra.items():
            if (
                name in extra_data_info_dict
                and extra_data_info_dict[name].sample_kind == "atom_pair"
            ):
                targets[name] = get_bidirectional_edges(tensor)

        return systems, targets, extra

    def reverse_transform(
        systems: list[System],
        targets: dict[str, TensorMap],
        extra: dict[str, TensorMap],
    ) -> tuple[list[System], dict[str, TensorMap], dict[str, TensorMap]]:
        """
        Transform function that gets the single direction edges from the
        bidirectional ones.

        :param systems: List of systems.
        :param targets: Dictionary containing the targets corresponding to the systems.
        :param extra: Dictionary containing any extra data.
        :return: The systems, targets and extra data with data on only one direction
          of the edges.
        """
        for name, tensor in targets.items():
            if (
                name in target_info_dict
                and target_info_dict[name].sample_kind == "atom_pair"
            ):
                targets[name] = get_single_direction_edges(tensor)

        for name, tensor in extra.items():
            if (
                name in extra_data_info_dict
                and extra_data_info_dict[name].sample_kind == "atom_pair"
            ):
                targets[name] = get_single_direction_edges(tensor)

        return systems, targets, extra

    return transform, reverse_transform


def get_pair_sample_labels(
    sample_labels: Labels,
    centers: Optional[torch.Tensor] = None,
    neighbors: Optional[torch.Tensor] = None,
    cell_shifts: Optional[torch.Tensor] = None,
    systems: Optional[List[System]] = None,
    nl_options: Optional[NeighborListOptions] = None,
) -> Labels:
    """
    Create per-pair sample labels from center and neighbor atom indices and cell shifts.

    Each row in the returned Labels corresponds to one directed edge (center → neighbor)
    in the neighbor list, identified in the same way as a standard metatensor neighbor
    list: by system index, center atom index, neighbor atom index, and the integer cell
    shift vector ``(cell_shift_a, cell_shift_b, cell_shift_c)``.

    Either ``centers``, ``neighbors`` and ``cell_shifts`` must all be provided directly
    (already flat and batch-offset, as e.g. produced internally by PET's own
    preprocessing), or ``systems`` and ``nl_options`` must be provided so that they can
    be computed by reading each system's own neighbor list.

    :param sample_labels: Labels for all atoms in the batch, with dimensions
        ``["system", "atom"]``, as returned by :func:`get_per_atom_sample_labels`.
    :param centers: Flat tensor of center atom global indices, shape ``(n_edges,)``.
    :param neighbors: Flat tensor of neighbor atom global indices, shape ``(n_edges,)``.
    :param cell_shifts: Integer cell shift vectors for each edge, shape ``(n_edges,
        3)``.
    :param systems: List of systems in the batch. Only used if ``centers``,
        ``neighbors`` and ``cell_shifts`` are not provided.
    :param nl_options: Options for the neighbor list used to enumerate edges. Only used
        if ``centers``, ``neighbors`` and ``cell_shifts`` are not provided.
    :return: Labels with columns ``["system", "first_atom", "second_atom",
        "cell_shift_a", "cell_shift_b", "cell_shift_c"]``, shape ``(n_edges, 6)``.
    """
    if centers is None or neighbors is None or cell_shifts is None:
        if systems is None or nl_options is None:
            raise ValueError(
                "either `centers`, `neighbors` and `cell_shifts` must all be "
                "provided, or `systems` and `nl_options` must be provided so "
                "they can be computed from the systems' neighbor lists"
            )
        centers, neighbors, cell_shifts = _pair_arrays_from_neighbor_lists(
            systems, nl_options
        )

    sample_values = sample_labels.values  # (n_atoms, 2): [system, atom]
    center_values = sample_values[centers]  # (n_edges, 2): [system, first_atom]
    neighbor_values = sample_values[neighbors]  # (n_edges, 2): [system, second_atom]

    pair_values = torch.cat(
        [
            center_values[:, :1],  # system        (n_edges, 1)
            center_values[:, 1:],  # first_atom    (n_edges, 1)
            neighbor_values[:, 1:],  # second_atom   (n_edges, 1)
            cell_shifts,  # a, b, c       (n_edges, 3)
        ],
        dim=1,
    )

    return Labels(
        names=[
            "system",
            "first_atom",
            "second_atom",
            "cell_shift_a",
            "cell_shift_b",
            "cell_shift_c",
        ],
        values=pair_values,
        assume_unique=True,
    )


def _pair_arrays_from_neighbor_lists(
    systems: List[System],
    nl_options: NeighborListOptions,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reads each system's own neighbor list and returns the flat, batch-offset
    ``centers``, ``neighbors`` and ``cell_shifts`` tensors required by
    :func:`get_pair_sample_labels`.

    :param systems: List of systems in the batch.
    :param nl_options: Options for the neighbor list used to enumerate edges.
    :return: Tuple of ``(centers, neighbors, cell_shifts)``.
    """
    device = systems[0].positions.device
    nl_values_list: List[torch.Tensor] = []
    num_edges: List[int] = []
    node_offsets_list: List[int] = []

    node_counter = 0
    for system in systems:
        assert len(system.known_neighbor_lists()) >= 1, "no neighbor list found"
        neighbor_list = system.get_neighbor_list(nl_options)
        nl_values = neighbor_list.samples.values
        nl_values_list.append(nl_values)

        system_size = len(system)
        node_offsets_list.append(node_counter)
        num_edges.append(nl_values.shape[0])
        node_counter += system_size

    nl_values = torch.cat(nl_values_list)
    centers = nl_values[:, 0]
    neighbors = nl_values[:, 1]
    cell_shifts = nl_values[:, 2:]

    # Compute the offsets
    total_edges = sum(num_edges)
    num_edges_tensor = torch.tensor(num_edges, device=device, dtype=torch.long)
    node_offsets = torch.tensor(node_offsets_list, device=device, dtype=torch.long)
    edge_offsets = torch.repeat_interleave(
        node_offsets, num_edges_tensor, output_size=total_edges
    ).to(dtype=centers.dtype)

    # Offset the centers and neighbors
    centers = centers + edge_offsets
    neighbors = neighbors + edge_offsets

    return centers, neighbors, cell_shifts


def _copy_block(block: TensorBlock) -> TensorBlock:
    return TensorBlock(
        values=block.values,
        samples=block.samples,
        components=block.components,
        properties=block.properties,
    )


def _lexsort_2col(values: torch.Tensor) -> torch.Tensor:
    """Argsort of a two-column int tensor, primarily by column 0 then column 1.
    Rows (property labels) are always unique, so no tie-breaking is needed."""
    if values.shape[0] == 0:
        return torch.arange(0)
    multiplier = int(values[:, 1].max().item()) + 1
    combined = values[:, 0].to(torch.int64) * multiplier + values[:, 1].to(torch.int64)
    return torch.argsort(combined)


def _reversed_atom_pair_block(
    block: TensorBlock,
    i_first: int,
    i_second: int,
    i_shift_a: int,
    i_shift_b: int,
    i_shift_c: int,
) -> TensorBlock:
    """Swap "first"/"second" atom throughout ``block``: negate the cell shift and
    swap ``first_atom``/``second_atom`` in the samples, transpose the two component
    axes, and re-sort the properties after swapping ``n_1``/``n_2`` (their swapped
    order isn't itself ascending)."""
    reversed_samples_values = block.samples.values.clone()
    reversed_samples_values[:, [i_first, i_second]] = reversed_samples_values[
        :, [i_second, i_first]
    ]
    reversed_samples_values[:, [i_shift_a, i_shift_b, i_shift_c]] *= -1
    reversed_samples = Labels(
        names=block.samples.names, values=reversed_samples_values
    )

    reversed_values = block.values.transpose(1, 2)
    # The component axes themselves were just swapped along with the values
    # (o3_mu_1 <-> o3_mu_2): axis 0 must always be named "o3_mu_1" and axis 1
    # "o3_mu_2" regardless of content, so re-tag (not just reorder) the two
    # component Labels rather than swapping the list order outright - which
    # would leave axis 0 named "o3_mu_2" whenever o3_lambda_1 != o3_lambda_2.
    reversed_components = [
        Labels(names=block.components[0].names, values=block.components[1].values),
        Labels(names=block.components[1].names, values=block.components[0].values),
    ]

    swapped_properties_values = block.properties.values[:, [1, 0]]
    order = _lexsort_2col(swapped_properties_values)
    reversed_properties = Labels(
        names=block.properties.names, values=swapped_properties_values[order]
    )
    reversed_values = reversed_values.index_select(-1, order)

    return TensorBlock(
        values=reversed_values,
        samples=reversed_samples,
        components=reversed_components,
        properties=reversed_properties,
    )


def expand_masked_atom_pair_samples(tmap: TensorMap) -> TensorMap:
    """
    Restores atom-pair data dropped by :class:`~metatrain.experimental
    .edge_composition.EdgeCompositionModel`'s upper-triangular masking in the
    uncoupled basis: within a same-atom-type-pair block, only the
    ``first_atom < second_atom`` samples are kept; for a cross-type pair, only
    one of the two type-ordered keys is produced at all (e.g. only H-O, not O-H).

    Both are the same symmetry: swapping "first"/"second" atom is equivalent to
    transposing the two component axes and swapping the ``n_1``/``n_2`` property
    axes of the *sibling* block (``o3_lambda_1``/``_2``, ``o3_sigma_1``/``_2`` and
    the two atom types all swapped) - applied at whichever granularity is
    missing, samples or the whole block. Verified exactly against real reference
    data, no extra sign. A tensor with neither form of masking passes through
    unchanged.

    Only the uncoupled basis is affected - the coupled basis already folds both
    directions into one block per unordered type pair (see
    :func:`get_bidirectional_edges`).

    :param tmap: An atom-pair TensorMap in the uncoupled basis
        (``"o3_lambda_1"``/``"o3_lambda_2"`` present in the keys).
    :return: ``tmap`` with masked same-type samples and missing cross-type keys
        restored; unchanged if neither form of masking applies.
    """
    key_names = tmap.keys.names
    if "first_atom_type" not in key_names or "second_atom_type" not in key_names:
        # Not an atom-pair tensor at all (e.g. a per-atom additive contribution) -
        # nothing to do.
        return tmap
    if "o3_lambda_1" not in key_names or "o3_lambda_2" not in key_names:
        # Coupled basis (or something else entirely) - not handled here, see
        # `get_bidirectional_edges` for the coupled case.
        return tmap

    i_z1 = key_names.index("first_atom_type")
    i_z2 = key_names.index("second_atom_type")
    i_l1 = key_names.index("o3_lambda_1")
    i_l2 = key_names.index("o3_lambda_2")
    i_s1 = key_names.index("o3_sigma_1")
    i_s2 = key_names.index("o3_sigma_2")

    i_first = tmap.block(0).samples.names.index("first_atom")
    i_second = tmap.block(0).samples.names.index("second_atom")
    i_shift_a = tmap.block(0).samples.names.index("cell_shift_a")
    i_shift_b = tmap.block(0).samples.names.index("cell_shift_b")
    i_shift_c = tmap.block(0).samples.names.index("cell_shift_c")

    new_keys_values: List[List[int]] = []
    new_blocks: List[TensorBlock] = []
    for key, block in tmap.items():
        key_values: List[int] = key.values.to(torch.int64).tolist()
        z1 = key_values[i_z1]
        z2 = key_values[i_z2]

        sibling_values: List[int] = key_values[:]
        sibling_values[i_l1] = key_values[i_l2]
        sibling_values[i_l2] = key_values[i_l1]
        sibling_values[i_s1] = key_values[i_s2]
        sibling_values[i_s2] = key_values[i_s1]
        sibling_values[i_z1] = key_values[i_z2]
        sibling_values[i_z2] = key_values[i_z1]

        sibling_pos: Optional[int] = tmap.keys.position(sibling_values)

        if z1 == z2:
            if sibling_pos is None:
                # No counterpart at all for this same-type key (e.g. no such
                # pairs occur in these systems in the first place) - nothing
                # to restore.
                new_keys_values.append(key_values)
                new_blocks.append(_copy_block(block))
            else:
                # Restore this block's missing first_atom > second_atom
                # samples using the sibling's first_atom < second_atom ones
                # (the sibling is this same block itself when o3_lambda_1 ==
                # o3_lambda_2 and o3_sigma_1 == o3_sigma_2).
                sibling_block = tmap.block_by_id(sibling_pos)
                reversed_sibling = _reversed_atom_pair_block(
                    sibling_block, i_first, i_second, i_shift_a, i_shift_b, i_shift_c
                )
                full_samples = Labels(
                    names=block.samples.names,
                    values=torch.concatenate(
                        [block.samples.values, reversed_sibling.samples.values],
                        dim=0,
                    ),
                )
                full_values = torch.concatenate(
                    [block.values, reversed_sibling.values], dim=0
                )
                new_keys_values.append(key_values)
                new_blocks.append(
                    TensorBlock(
                        values=full_values,
                        samples=full_samples,
                        components=block.components,
                        properties=block.properties,
                    )
                )
        else:
            new_keys_values.append(key_values)
            new_blocks.append(_copy_block(block))
            if sibling_pos is None:
                # The reverse-type-ordered key is entirely missing - this is
                # the only direction present for this pair of types, so
                # synthesize the missing sibling block wholesale from it.
                new_keys_values.append(sibling_values)
                new_blocks.append(
                    _reversed_atom_pair_block(
                        block, i_first, i_second, i_shift_a, i_shift_b, i_shift_c
                    )
                )
            # else: both directions are already present as separate,
            # presumably already-complete blocks (e.g. a `graph2mat`-style
            # additive model that never masks cross-type pairs) - leave both
            # unchanged; the sibling key will be visited on its own turn in
            # this same loop.

    new_keys = Labels(
        names=key_names,
        values=torch.tensor(new_keys_values, dtype=tmap.keys.values.dtype),
    )
    return TensorMap(keys=new_keys, blocks=new_blocks)
