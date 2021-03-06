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
        self.grad: Optional['Tensor'] = None

        if self.requires_grad:
            self.zero_grad()

    def zero_grad(self) -> None:
        """Zero out the gradient"""
        self.grad = Tensor(np.zeros_like(self.data))

    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def backward(self, grad: 'Tensor' = None) -> None:
        assert self.requires_grad, "Called backward on non-requires-grad tensor"

        if grad is None:
            if self.shape == ():
                grad = Tensor(1)
            else:
                raise RuntimeError("grad must be specified for nonzero tensor.")

        self.grad.data += grad.data

        for dependency in self.depends_on:
            backward_grad = dependency.grad_fn(grad.data)
            dependency.tensor.backward(Tensor(backward_grad))

    def sum(self) -> 'Tensor':
        return tensor_sum(self)


def tensor_sum(t: Tensor) -> Tensor:
    r"""Sum a tensor

    Args:
        t: tensor to be summed

    Returns:
        the sum, zero-dimensional tensor
    """
    data = t.data.sum()
    requires_grad = t.requires_grad

    if requires_grad:

        def grad_fn(grad: np.ndarray) -> np.ndarray:
            r"""The gradient is a 0-tensor

            Each element contributes a gradient of 0-tensor (1).
            """
            return grad * np.ones_like(t.data)

        depends_on = [Dependency(t, grad_fn)]

    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)


def ensure_array(arrayable: Arrayable) -> np.ndarray:
    r"""Ensure arrayable type is an array"""
    if isinstance(arrayable, np.ndarray):
        return arrayable
    else:
        return np.array(arrayable)
