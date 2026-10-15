r"""
Scaler
======

The scaler is a simple model that computes per-target and per-property scaling factors.
It is meant to be used as a preprocessing step for other architectures, so that targets
are standardized before being fed to the main model.

See :ref:`scale-targets` for more details.
"""

from typing import Dict, Literal, Optional, Union

from typing_extensions import NotRequired, TypedDict


FixedScalerWeights = Dict[str, Union[float, Dict[int, float]]]


class ModelHypers(TypedDict):
    """Hyperparameters for the scaler."""

    densify_atomic_basis: bool = True
    """Whether to densify the atomic basis targets when computing the scaling
    weights. This can only be done if the target is loaded from a DiskDataset.

    Most models will require the scaler to work with the densified atomic basis.
    """
    use_onsite_scales_for_offsite: bool = False
    """Use onsite (node) scales as a proxy for offsite (edge) scales.

    Atom-pair (edge) targets are normally excluded from the scaler entirely and
    keep the identity scale of 1.0. When this is ``True`` they are kept, and
    after fitting the per-target scale of every ``mtt::matrix_edges::X`` target
    whose matching ``mtt::matrix_nodes::X`` target is also fitted is replaced by
    a mean of the two node scales for each atomic-type pair (see
    :attr:`offsite_proxy_mean`).

    Edge targets are dominated by near-zero couplings, so a scale measured from
    the edge data itself would be small and give the edge target little weight
    in the loss. The alternative is to subtract a learned per-element-pair
    baseline first, with an edge composition model.
    """
    offsite_proxy_mean: Literal["geometric", "arithmetic"] = "geometric"
    """Aggregation used for the onsite-scale proxy.

    ``"geometric"`` uses ``sqrt(s_I * s_J)``, ``"arithmetic"`` uses
    ``(s_I + s_J) / 2``. The geometric mean is the AM-GM lower bound, so it
    gives smaller proxy scales and a higher relative edge loss weight. Only
    used when :attr:`use_onsite_scales_for_offsite` is ``True``.
    """


class TrainerHypers(TypedDict):
    """Hyperparameters for the scaler trainer."""

    fixed_weights: FixedScalerWeights = {}
    """Weights for target scaling.

    This is passed to the ``fixed_weights`` argument of
    :meth:`Scaler.train_model <metatrain.scaler.Scaler.train_model>`,
    see its documentation to understand exactly what to pass here.
    """
    additive_models: list[str] = []
    """List of checkpoint files to load additive models from.

    The contribution from these models will be subtracted from the targets
    before computing the scales.
    """
    batch_size: Optional[int] = None
    """Number of structures to accumulate at a time.
    This only affects memory usage, not the resulting scales, since the
    scaler is a deterministic modelrather than an iterative optimization.
    Defaults to the size of the smallest training dataset.
    """
    per_structure_targets: list[str] = []
    """Target names that should be treated as
    per-structure quantities and therefore not divided by the number of atoms.
    """
    distributed: NotRequired[bool]
    """Whether to use distributed training. When not set, distributed training
    is enabled automatically when running under more than one SLURM task.
    Setting this option explicitly is deprecated."""
    distributed_port: int = 39591
    """Port for distributed communication among processes"""
    num_workers: Optional[int] = None
    """Number of workers for data loading. If not provided, it is set
    automatically."""

