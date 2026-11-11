from types import SimpleNamespace

import torch

from imagehub.backends.nucleus_image import encode_prompt


class _Tokenizer:
    def __init__(self) -> None:
        self.padded_length = 0

    def tokenize(self, text: str) -> list[int]:
        assert text.startswith("<|im_start|>system\n")
        return list(range(10))

    def __call__(self, text: str, max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        del text
        self.padded_length = max_length
        ids = torch.zeros(1, max_length, dtype=torch.long)
        mask = torch.arange(max_length).unsqueeze(0) < 10
        return ids, mask


class _TextEncoder:
    def __call__(self, ids, mask, *, position_ids, output_hidden_states):
        assert ids.shape == mask.shape == (1, 16)
        assert position_ids.shape == (3, 1, 16)
        assert output_hidden_states is True
        return [torch.zeros(1, 16, 2) for _ in range(8)]


def test_encode_prompt_pads_to_the_next_multiple_of_eight() -> None:
    tokenizer = _Tokenizer()
    components = SimpleNamespace(tokenizer=tokenizer, text_encoder=_TextEncoder())
    states, mask = encode_prompt(components, "a cat", torch.device("cpu"), 1024)
    assert tokenizer.padded_length == 16
    assert states.shape == (1, 16, 2)
    assert mask is not None and mask.shape == (1, 16)
