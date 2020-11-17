import pytest

from autograd.tensor import Tensor


def test_integer_grad(integer):
    t = _sum_tensor(integer)
    assert t.grad.data.item() == 1


def test_array_grad(array):
    t = _sum_tensor(array)
    ones = [1 for x in range(len(array))]
    assert (t.grad.data == ones).all()


def _sum_tensor(data):
    t1 = Tensor(data, requires_grad=True)
    t2 = t1.sum()
    t2.backward()
    return t1

