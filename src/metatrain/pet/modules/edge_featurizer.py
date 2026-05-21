"""
Module to build edge features directly from node features.

For ``atom_pair`` targets (e.g. the off-diagonal blocks of a Hamiltonian
matrix), the edge features can be computed from the node features of the two
atoms forming each edge, instead of being read out from the edge transformer.
Since the edge transformer scales quadratically with the number of edges per
node, this allows atom-pair edges to be enumerated at a larger cutoff than the
one used by the GNN, without paying the quadratic cost.
"""

from typing import List

import torch
from torch import nn


class EdgeFeaturizerFromNodes(nn.Module):
    """
    Builds edge features from the node features of the two atoms of each edge.

    For each directed edge ``i -> j`` and each readout layer, the edge feature is
    ``MLP(concat(node_i, node_j, embedding(edge_vector_ij)))``. The construction
    is directional: edge ``i -> j`` and edge ``j -> i`` differ both in the node
    order and in the edge vector (``r_ij = -r_ji``).

    :param num_layers: Number of readout layers (one MLP is used per layer).
    :param d_node: Dimension of the node features.
    :param d_pet: Dimension of the produced edge features (and of the edge
        vector embedding).
    """

    def __init__(self, num_layers: int, d_node: int, d_pet: int) -> None:
        super().__init__()
        self.edge_vector_embedder = nn.Linear(4, d_pet)
        self.mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * d_node + d_pet, 2 * d_pet),
                    nn.SiLU(),
                    nn.Linear(2 * d_pet, d_pet),
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        node_features_list: List[torch.Tensor],
        centers: torch.Tensor,
        neighbors: torch.Tensor,
        edge_vectors: torch.Tensor,
        edge_distances: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        :param node_features_list: List of node feature tensors, one per readout
            layer, each of shape ``(n_nodes, d_node)``.
        :param centers: Global atom indices of the center atom of each edge,
            shape ``(n_edges,)``.
        :param neighbors: Global atom indices of the neighbor atom of each edge,
            shape ``(n_edges,)``.
        :param edge_vectors: Cartesian edge vectors, shape ``(n_edges, 3)``.
        :param edge_distances: Edge distances, shape ``(n_edges,)``.
        :return: List of edge feature tensors, one per readout layer, each of
            shape ``(n_edges, d_pet)``.
        """
        edge_vector_embedding = self.edge_vector_embedder(
            torch.cat([edge_vectors, edge_distances[:, None]], dim=-1)
        )

        edge_features_list: List[torch.Tensor] = []
        for i, mlp in enumerate(self.mlps):
            node_features = node_features_list[i]
            edge_input = torch.cat(
                [
                    node_features[centers],
                    node_features[neighbors],
                    edge_vector_embedding,
                ],
                dim=-1,
            )
            edge_features_list.append(mlp(edge_input))
        return edge_features_list


class DummyEdgeFeaturizer(nn.Module):
    """Dummy module to keep TorchScript happy when edge features are not built
    from node features. It is never executed at runtime."""

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        node_features_list: List[torch.Tensor],
        centers: torch.Tensor,
        neighbors: torch.Tensor,
        edge_vectors: torch.Tensor,
        edge_distances: torch.Tensor,
    ) -> List[torch.Tensor]:
        empty: List[torch.Tensor] = []
        return empty
