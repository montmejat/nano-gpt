import argparse

import numpy as np
from dataset import TinyShakespeare

from tinygrad import Tensor, nn


class Bigram:
    def __init__(self, vocab_size: int):
        self.embedding = nn.Embedding(vocab_size=vocab_size, embed_size=vocab_size)

    def __call__(self, x: Tensor):
        return self.embedding(x)

    @property
    def weights(self):
        return [self.embedding.weight]

    def yap(self, first_token: int, length: int):
        token = first_token
        tokens = []

        for _ in range(length):
            x = Tensor(float(token))
            out = self(x)
            out = out[:, :].squeeze()  # Get last token
            out = out.softmax()  # Get probabilities
            # Sample a random token from the distribution
            token = np.random.multinomial(1, out.numpy()).argmax()
            tokens.append(int(token))

        return tokens


def estimate_loss(model: Bigram, steps: int = 300):
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
    parser.add_argument("--steps", type=int, default=3000)
    args = parser.parse_args()

    model = Bigram(len(TinyShakespeare._vocab))
    optimizer = nn.optim.AdamW(model.weights, lr=args.learning_rate)

    with Tensor.train():
        for i in range(args.steps):
            if i % 300 == 0:
                losses = estimate_loss(model)
                print(f"Step {i}: {losses}")

            x, y = TinyShakespeare.get_random_batch("train", batch_size=args.batch_size)

            logits = model(x)
            loss = logits.sparse_categorical_crossentropy(y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
