import pytest

from autograd.tensor import Tensor


@pytest.fixture(params=[0, 1, 2])
def integer(request):
    return request.param


@pytest.fixture(params=[[0, 1], [1, 2, 3]])
def array(request):
    return request.param
