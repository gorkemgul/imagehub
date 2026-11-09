import shutil

from imagehub.tokenization_byt5 import ByT5Tokenizer


def test_byt5_existing_tokenize_interface_is_unchanged(tmp_path):
    source = "tests/fixtures/glm_image_configs/tokenizer_tokenizer_config.json"
    shutil.copyfile(source, tmp_path / "tokenizer_config.json")
    tokenizer = ByT5Tokenizer(tmp_path)

    assert tokenizer.tokenize("") == [1]
    assert tokenizer.tokenize("aé") == [100, 198, 172, 1]
    assert tokenizer.tokenize("abcdef", 4) == [100, 101, 102, 1]
