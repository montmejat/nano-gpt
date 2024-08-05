import os
from random import randint
from typing import Any

import torch

data = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
if not os.path.exists("../input.txt"):
    os.system(f"wget {data} -O ../input.txt")

input = open("../input.txt").read()
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

        cls._train_tokens = torch.tensor(cls._input_tokens[:split_idx])
        cls._val_tokens = torch.tensor(cls._input_tokens[split_idx:])

    @classmethod
    def to(cls: "TinyShakespeare", device: Any):
        cls._train_tokens = cls._train_tokens.to(device)
        cls._val_tokens = cls._val_tokens.to(device)

    @classmethod
    def get_random_batch(
        cls: "TinyShakespeare", set: str, batch_size: int = 8, sequence_length: int = 1
    ):
        tokens = cls._train_tokens if set == "train" else cls._val_tokens
        indexes = [
            randint(0, len(tokens) - sequence_length - 1) for _ in range(batch_size)
        ]
        x = torch.stack([tokens[i : i + sequence_length] for i in indexes])
        y = torch.stack([tokens[i + 1 : i + sequence_length + 1] for i in indexes])

        if sequence_length == 1:
            x = x.squeeze()
            y = y.squeeze()

        return x, y


TinyShakespeare.split()
