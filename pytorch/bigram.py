import argparse

import torch
from torch import Tensor, nn, optim
from torch.nn import functional as F

from dataset import TinyShakespeare


class Bigram(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x: Tensor):
        logits = self.token_embedding(x)
        return logits

    def generate(self, x: Tensor, length: int):
        tokens = x

        for _ in range(length):
            x = tokens[:, -1]
            logits = self(x)
            probs = F.softmax(logits, dim=-1)
            pred = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat((tokens, pred), dim=1)

        return tokens


@torch.no_grad()
def estimate_loss(model: Bigram, steps: int = 300):
    out = {}

    for split in ["train", "val"]:
        losses = []

        for _ in range(steps):
            x, y = TinyShakespeare.get_random_batch(split)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
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
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    for i in range(args.steps):
        if i % 300 == 0:
            model.eval()
            losses = estimate_loss(model)
            print(f"Step {i}: {losses}")
            model.train()

        x, y = TinyShakespeare.get_random_batch("train", batch_size=args.batch_size)

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    model.eval()
    tokens = model.generate(torch.tensor([[0]]), length=500)
    for token in tokens[0]:
        print(TinyShakespeare._token_to_char[token.item()], end="")
    print()