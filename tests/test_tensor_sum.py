import pytest

from autograd.tensor import Tensor


def test_integer_grad(integer):
    t1 = Tensor(integer, requires_grad=True)
    t2 = t1.sum()
    t2.backward()
    assert t1.grad.data.item() == 1

