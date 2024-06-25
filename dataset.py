import os
from random import randint

from tinygrad import Tensor

data = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
if not os.path.exists("input.txt"):
    os.system(f"wget {data}")

input = open("input.txt").read()
vocab = sorted(set("".join(input)))
char_to_token = {c: i for i, c in enumerate(vocab)}
token_to_char = {i: c for i, c in enumerate(vocab)}
input_tokens = [char_to_token[c] for c in input]


class TinyShakespeare:
    _input = input
    _vocab = vocab
    _char_to_token = char_to_token
    _token_to_char = token_to_char
    _input_tokens = input_tokens

    @classmethod
    def split(cls: "TinyShakespeare", train: float = 0.9, val: float = 0.1):
        split_idx = int(len(cls._input) * train)

        # Make sure we don't split in the middle of a line
        while True:
            if cls._input_tokens[split_idx] == cls._char_to_token["\n"]:
                break
            split_idx += 1

        cls._train_tokens = cls._input_tokens[:split_idx]
        cls._val_tokens = cls._input_tokens[split_idx:]

    @classmethod
    def get_random_batch(cls: "TinyShakespeare", set: str, batch_size: int = 8):
        tokens = cls._train_tokens if set == "train" else cls._val_tokens
        indexes = [randint(0, len(tokens)) for _ in range(batch_size)]
        x = Tensor([tokens[i] for i in indexes])
        y = Tensor([tokens[i + 1] for i in indexes])
        return x, y


TinyShakespeare.split()
