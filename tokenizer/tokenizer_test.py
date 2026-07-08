import pytest
from tokenizer.tokenizer import Tokenizer


@pytest.fixture
def tokenizer() -> Tokenizer:
    return Tokenizer()


class TestTokenize:
    def test_splits_into_characters(self, tokenizer: Tokenizer) -> None:
        assert tokenizer.tokenize("hello") == ["h", "e", "l", "l", "o"]

    def test_removes_spaces(self, tokenizer: Tokenizer) -> None:
        assert tokenizer.tokenize("hello world") == [
            "h",
            "e",
            "l",
            "l",
            "o",
            "w",
            "o",
            "r",
            "l",
            "d",
        ]

    def test_empty_string(self, tokenizer: Tokenizer) -> None:
        assert tokenizer.tokenize("") == []

    def test_multiple_spaces(self, tokenizer: Tokenizer) -> None:
        assert tokenizer.tokenize("a   b") == ["a", "b"]

    def test_leading_trailing_spaces(self, tokenizer: Tokenizer) -> None:
        assert tokenizer.tokenize("  hi  ") == ["h", "i"]

    def test_special_characters(self, tokenizer: Tokenizer) -> None:
        assert tokenizer.tokenize("a, b!") == ["a", ",", "b", "!"]

    def test_numbers(self, tokenizer: Tokenizer) -> None:
        assert tokenizer.tokenize("42") == ["4", "2"]


class TestBuildVocab:
    def test_builds_vocab_from_text(self, tokenizer: Tokenizer) -> None:
        tokenizer.build_vocab("abc")
        assert tokenizer.get_vocab() == {"a": 0, "b": 1, "c": 2}

    def test_duplicate_chars_get_last_index(self, tokenizer: Tokenizer) -> None:
        tokenizer.build_vocab("aba")
        assert tokenizer.get_vocab() == {"a": 2, "b": 1}

    def test_empty_text(self, tokenizer: Tokenizer) -> None:
        tokenizer.build_vocab("")
        assert tokenizer.get_vocab() == {}

    def test_ignores_spaces(self, tokenizer: Tokenizer) -> None:
        tokenizer.build_vocab("a b c")
        assert tokenizer.get_vocab() == {"a": 0, "b": 1, "c": 2}


class TestEncode:
    def test_encode_returns_vocab_values(self, tokenizer: Tokenizer) -> None:
        tokenizer.build_vocab("abc")
        assert sorted(tokenizer.encode_vocab()) == [0, 1, 2]

    def test_encode_empty_vocab(self, tokenizer: Tokenizer) -> None:
        assert tokenizer.encode_vocab() == []


class TestDecode:
    def test_decodes_token_ids_to_string(self, tokenizer: Tokenizer) -> None:
        tokenizer.build_vocab("cat dog bird")
        assert tokenizer.decode([0, 1, 2]) == "cat"

    def test_unknown_token_id_raises_error(self, tokenizer: Tokenizer) -> None:
        tokenizer.build_vocab("ab")
        with pytest.raises(ValueError, match="Unkwon token_id : 99"):
            tokenizer.decode([99])

    def test_empty_list_returns_empty_string(self, tokenizer: Tokenizer) -> None:
        tokenizer.build_vocab("ab")
        assert tokenizer.decode([]) == ""
