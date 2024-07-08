import argparse

import numpy as np
from dataset import TinyShakespeare

from tinygrad import Tensor, nn


class SelfAttentionHead:
    """
    This class represents a single head in the transformer model.

    Args:
        sequence_length (int): The length of the input sequence.
        embed_size (int): The size of the input embeddings.
        head_size (int): The size of the head.
    """

    def __init__(self, sequence_length: int, embed_size: int, head_size: int):
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)
        self.embed_size = embed_size

        inf = Tensor.full((sequence_length, sequence_length), float("-inf"))
        self.inf_tri = inf.triu(1)

    def __call__(self, x: Tensor):
        """
        x: Tensor of size (batch_size, sequence_length, embed_size).
        """

        key = self.key(x)
        query = self.query(x)

        wei_mat = query @ key.transpose(-2, -1) * key.shape[-1] ** (-0.5)
        wei_mat = (wei_mat + self.inf_tri[: x.shape[1], : x.shape[1]]).softmax(-1)

        value = self.value(x)
        return wei_mat @ value

    @property
    def weights(self):
        return [self.key.weight, self.query.weight, self.value.weight]


class Transfomer:
    def __init__(self, vocab_size: int, sequence_length: int, embed_size: int):
        self.sequence_length = sequence_length
        self.embed_size = embed_size

        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_embedding = nn.Embedding(sequence_length, embed_size)
        self.self_att = SelfAttentionHead(
            sequence_length, embed_size, head_size=embed_size
        )
        self.decoder = nn.Linear(embed_size, vocab_size)

    def __call__(self, x: Tensor):
        token_embeddings = self.token_embedding(x)
        posit_embeddings = self.positional_embedding(Tensor.arange(x.shape[1]))
        x = self.self_att(token_embeddings + posit_embeddings)
        logits = self.decoder(x)
        return logits

    @property
    def weights(self):
        return [
            self.token_embedding.weight,
            self.positional_embedding.weight,
            self.decoder.weight,
        ] + self.self_att.weights

    def yap(self, first_token: Tensor, length: int):
        tokens = first_token

        for _ in range(length):
            x = tokens[:, -self.sequence_length :]
            out = self(x)
            out = out[:, -1, :].squeeze()  # Get last token
            out = out.softmax()  # Get probabilities
            # Sample a random token from the distribution
            token = np.random.multinomial(1, out.numpy()).argmax()
            tokens = tokens.cat(Tensor([[token]]), dim=1)

        return tokens


def estimate_loss(model: Transfomer, steps: int = 300):
    out = {}

    for split in ["train", "val"]:
        losses = []

        for _ in range(steps):
            x, y = TinyShakespeare.get_random_batch(split)
            logits = model(x)
            loss = logits.sparse_categorical_crossentropy(y)
            losses.append(loss.item())

        out[split] = sum(losses) / len(losses)

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=int, default=0.01)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--embed-size", type=int, default=16)
    args = parser.parse_args()

    model = Transfomer(
        len(TinyShakespeare._vocab), args.sequence_length, args.embed_size
    )
    optimizer = nn.optim.AdamW(model.weights, lr=args.learning_rate)

    with Tensor.train():
        for i in range(args.steps):
            if i % 10 == 0:
                losses = estimate_loss(model)
                print(f"Step {i}: {losses}")

            x, y = TinyShakespeare.get_random_batch(
                "train", args.batch_size, args.sequence_length
            )

            logits = model(x)
            loss = logits.sparse_categorical_crossentropy(y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    tokens = model.yap(Tensor([[0]]), 100)
    for token in tokens[0]:
        print(TinyShakespeare._token_to_char[token.item()], end="")
    print()
