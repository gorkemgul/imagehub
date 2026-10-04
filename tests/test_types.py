import pytest
import torch

from imagehub.image import Image
from imagehub.types import ImageResult, T2IRequest, Task


def test_task_values():
    assert Task("t2i") is Task.T2I


def test_request_validation_and_copy():
    params = {"seed": 1}
    request = T2IRequest("a cat", params)
    params["seed"] = 2
    assert request.parameters == {"seed": 1}
    with pytest.raises(ValueError):
        T2IRequest("   ")


def test_result_validation():
    image = Image(torch.zeros(2, 2, 3, dtype=torch.uint8))
    result = ImageResult([image], {"seed": 3})
    assert result.images == (image,)
    with pytest.raises(ValueError):
        ImageResult([])
    with pytest.raises(ValueError):
        ImageResult([torch.zeros(2, 2, 3)])
