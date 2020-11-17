import numpy as np

from typing import List, NamedTuple, Callable, Optional, Union


class Dependency(NamedTuple):
    tensor: "Tensor"
    grad_fn: Callable[[np.ndarray], np.ndarray]


Arrayable = Union[float, list, np.ndarray]


class Tensor:
    """Tensor

    Stores data as an np.ndarray, and maintains
    both the gradient with respect to that data,
    and its dependencies in the computation graph.
    """

    def __init__(
        self,
        data: Arrayable,
        requires_grad: bool = False,
        depends_on: List[Dependency] = None,
    ) -> None:
        self.data = ensure_array(data)
        self.requires_grad = requires_grad
        self.depends_on = depends_on or []
        self.shape = self.data.shape
        self.grad: Optional["Tensor"] = None

        if self.requires_grad:
            self.zero_grad()

    def zero_grad(self) -> None:
        """Zero out the gradient"""
        self.grad = Tensor(np.zeros_like(self.data))

    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"


def ensure_array(arrayable: Arrayable) -> np.ndarray:
    r"""Ensure arrayable type is an array"""
    if isinstance(arrayable, np.ndarray):
        return arrayable
    else:
        return np.array(arrayable)
