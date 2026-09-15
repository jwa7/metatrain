import torch


def cutoff_func_bump(
    values: torch.Tensor, cutoff: torch.Tensor, width: float
) -> torch.Tensor:
    """
    Bump cutoff function.

    The 1e-6 margin from the boundary avoids numerical instabilities
    in the region where the functions transitions with C^inf smoothness
    from 1 and to 0.

    :param values: Distances at which to evaluate the cutoff function.
    :param cutoff: Cutoff radius for each node.
    :param width: Width of the cutoff region.
    :return: Values of the cutoff function at the specified distances.
    """

    scaled_values = (values - (cutoff - width)) / width
    clamped = scaled_values.clamp(1e-6, 1.0 - 1e-6)
    return 0.5 * (1 + torch.tanh(1 / torch.tan(torch.pi * clamped)))


def cutoff_func_cosine(
    values: torch.Tensor, cutoff: torch.Tensor, width: float
) -> torch.Tensor:
    """
    Cosine cutoff function.

    :param values: Distances at which to evaluate the cutoff function.
    :param cutoff: Cutoff radius for each node.
    :param width: Width of the cutoff region.
    :return: Values of the cutoff function at the specified distances.
    """

    scaled_values = (values - (cutoff - width)) / width
    clamped = scaled_values.clamp(0.0, 1.0)
    return 0.5 * (1 + torch.cos(torch.pi * clamped))


def apply_cutoff_function(
    values: torch.Tensor, cutoff: torch.Tensor, width: float, cutoff_function: str
) -> torch.Tensor:
    """
    Dispatch to :func:`cutoff_func_bump`/:func:`cutoff_func_cosine` by name, matching
    the ``cutoff_function`` hyper's own dispatch in
    ``structures.compute_batch_tensors``. Factored out for reuse by code that needs
    to evaluate the envelope at a *different* cutoff than the model's own (e.g. an
    outer cutoff), where reimplementing the dispatch inline would just duplicate it.

    :param values: Distances at which to evaluate the cutoff function.
    :param cutoff: Cutoff radius (per-edge tensor, or a scalar tensor broadcasting
        to all edges).
    :param width: Width of the cutoff region.
    :param cutoff_function: Either ``"bump"`` or ``"cosine"`` (case-insensitive).
    :return: Values of the cutoff function at the specified distances.
    """
    if cutoff_function.lower() == "bump":
        return cutoff_func_bump(values, cutoff, width)
    elif cutoff_function.lower() == "cosine":
        return cutoff_func_cosine(values, cutoff, width)
    else:
        raise ValueError(
            f"Unknown cutoff function type: {cutoff_function}. "
            f"Supported types are 'Cosine' and 'Bump'."
        )


class DummyModule(torch.nn.Module):
    """Dummy torch module to make torchscript happy.
    This model should never be run"""

    def __init__(self) -> None:
        super(DummyModule, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("This model should never be run")
