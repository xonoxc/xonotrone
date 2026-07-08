import pytest
from tokenizer.tokenizer import Tokenizer, Vocab, ReverseVocab


@pytest.fixture
def tokenizer() -> Tokenizer:
    return Tokenizer()


class TestTokenizer:
    def test_splits_on_spaces(self, tokenizer: Tokenizer) -> None:
        assert tokenizer.tokenize("hello world") == ["hello", "world"]

    def test_empty_string(self, tokenizer: Tokenizer) -> None:
        assert tokenizer.tokenize("") == [""]

    def test_single_word(self, tokenizer: Tokenizer) -> None:
        assert tokenizer.tokenize("hello") == ["hello"]

    def test_multiple_spaces(self, tokenizer: Tokenizer) -> None:
        assert tokenizer.tokenize("hello   world") == ["hello", "", "", "world"]

    def test_leading_trailing_spaces(self, tokenizer: Tokenizer) -> None:
        assert tokenizer.tokenize("  hello world  ") == ["", "", "hello", "world", "", ""]

    def test_special_characters(self, tokenizer: Tokenizer) -> None:
        assert tokenizer.tokenize("hello, world!") == ["hello,", "world!"]

    def test_numbers(self, tokenizer: Tokenizer) -> None:
        assert tokenizer.tokenize("42 hello 7") == ["42", "hello", "7"]


class TestBuildVocab:
    def test_builds_vocab_from_text(self, tokenizer: Tokenizer) -> None:
        tokenizer.build_vocab("cat dog cat bird")
        assert tokenizer.get_vocab() == {"cat": 2, "dog": 1, "bird": 3}

    def test_empty_text(self, tokenizer: Tokenizer) -> None:
        tokenizer.build_vocab("")
        assert tokenizer.get_vocab() == {"": 0}

    def test_single_word(self, tokenizer: Tokenizer) -> None:
        tokenizer.build_vocab("hello")
        assert tokenizer.get_vocab() == {"hello": 0}


class TestEncode:
    def test_encode_returns_vocab_values(self, tokenizer: Tokenizer) -> None:
        tokenizer.build_vocab("a b c")
        assert sorted(tokenizer.encode()) == [0, 1, 2]

    def test_encode_empty_vocab(self, tokenizer: Tokenizer) -> None:
        assert tokenizer.encode() == []


class TestDecode:
    def test_decodes_token_ids_to_string(self, tokenizer: Tokenizer) -> None:
        tokenizer.build_vocab("cat dog bird")
        assert tokenizer.decode([0, 1, 2]) == "cat dog bird"

    def test_unknown_token_id_raises_error(self, tokenizer: Tokenizer) -> None:
        tokenizer.build_vocab("cat dog")
        with pytest.raises(ValueError, match="Unkwon token_id : 99"):
            tokenizer.decode([99])

    def test_empty_list_returns_empty_string(self, tokenizer: Tokenizer) -> None:
        tokenizer.build_vocab("cat dog")
        assert tokenizer.decode([]) == ""
