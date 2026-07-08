type Vocab = dict[str, int]
type ReverseVocab = dict[int, str]


class Tokenizer:
    def __init__(self) -> None:
        self.vocab: Vocab = {}
        self.reverse_vocab: ReverseVocab = {}

    def build_vocab(
        self,
        text: str,
    ) -> None:
        tokenized_text = self.tokenize(text)

        for idx, token in enumerate(tokenized_text):
            self.vocab[token] = idx
            self.reverse_vocab[idx] = token

    def tokenize(self, txt: str) -> list[str]:
        return txt.split(" ")

    def get_vocab(self) -> Vocab:
        return self.vocab

    def encode(self) -> list[int]:
        return [value for value in self.vocab.values()]

    def decode(self, token_ids: list[int]) -> str:
        tokens = []
        for token_id in token_ids:
            if token_id not in self.reverse_vocab:
                raise ValueError(f"Unkwon token_id : {token_id}")

            tokens.append(
                self.reverse_vocab[token_id],
            )

        return " ".join(tokens)
