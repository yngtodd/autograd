import pytest

from autograd.tensor import Tensor


def test_integer_grad(integer):
    t1 = Tensor(integer, requires_grad=True)
    t2 = t1.sum()
    t2.backward()
    assert t1.grad.data.item() == 1


def test_array_grad(array):
    t1 = Tensor(array, requires_grad=True)
    t2 = t1.sum()
    t2.backward()
    ones = [1 for x in range(len(array))]
    assert (t1.grad.data == ones).all()
