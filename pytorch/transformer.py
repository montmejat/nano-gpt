import argparse

import torch
from dataset import TinyShakespeare
from torch import Tensor, nn, optim, tensor
from torch.nn import functional as F


class SelfAttentionHead(nn.Module):
    """
    This class represents a single head in the transformer model.

    Args:
        sequence_length (int): The length of the input sequence.
        embed_size (int): The size of the input embeddings.
        head_size (int): The size of the head.
    """

    def __init__(
        self,
        sequence_length: int,
        embed_size: int,
        head_size: int,
        dropout: float,
    ):
        super().__init__()

        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)
        self.embed_size = embed_size

        inf = torch.full((sequence_length, sequence_length), float("-inf"))
        self.inf_tri = inf.triu(1)

        self.dropout = nn.Dropout(dropout)

    def __call__(self, x: Tensor):
        """
        x: Tensor of size (batch_size, sequence_length, embed_size).
        """

        key = self.key(x)
        query = self.query(x)

        wei_mat = query @ key.transpose(-2, -1) * key.shape[-1] ** (-0.5)
        wei_mat = (wei_mat + self.inf_tri[: x.shape[1], : x.shape[1]]).softmax(-1)
        wei_mat = self.dropout(wei_mat)

        value = self.value(x)
        return wei_mat @ value


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        sequence_length: int,
        embed_size: int,
        head_size: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()

        self.heads = nn.ModuleList(
            [
                SelfAttentionHead(sequence_length, embed_size, head_size, dropout)
                for _ in range(num_heads)
            ]
        )
        self.proj = nn.Linear(embed_size, embed_size)

    def __call__(self, x: Tensor):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, embed_size: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor):
        return self.net(x)


class Block(nn.Module):
    def __init__(
        self,
        sequence_length: int,
        embed_size: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()

        self.self_att = MultiHeadSelfAttention(
            sequence_length,
            embed_size,
            head_size=embed_size // num_heads,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.feed_forward = FeedForward(embed_size, dropout)

        self.lnorm1 = nn.LayerNorm(embed_size)
        self.lnorm2 = nn.LayerNorm(embed_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor):
        x = x + self.self_att(self.lnorm1(x))
        x = x + self.feed_forward(self.lnorm2(x))
        x = self.dropout(x)
        return x


class Transfomer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        sequence_length: int,
        embed_size: int,
        num_heads: int,
        num_blocks: int,
        dropout: float,
    ):
        super().__init__()

        self.sequence_length = sequence_length
        self.embed_size = embed_size

        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_embedding = nn.Embedding(sequence_length, embed_size)

        self.blocks = nn.Sequential(
            *[
                Block(sequence_length, embed_size, num_heads, dropout)
                for _ in range(num_blocks)
            ]
        )
        self.layer_norm = nn.LayerNorm(embed_size)

        self.decoder = nn.Linear(embed_size, vocab_size)

    def __call__(self, x: Tensor):
        token_embeddings = self.token_embedding(x)
        posit_embeddings = self.positional_embedding(torch.arange(x.shape[1]))
        x = self.blocks(token_embeddings + posit_embeddings)
        x = self.layer_norm(x)
        logits = self.decoder(x)
        return logits

    def yap(self, first_token: Tensor, length: int):
        tokens = first_token

        for _ in range(length):
            x = tokens[:, -self.sequence_length :]
            logits = self(x)[:, -1, :]
            probs = F.softmax(logits, dim=-1).squeeze(0).squeeze(0)
            token = torch.multinomial(probs, num_samples=1).unsqueeze(0)
            tokens = torch.cat((tokens, token), dim=1)

        return tokens


@torch.no_grad()
def estimate_loss(model: Transfomer, sequence_length: int, steps: int = 300):
    out = {}

    for split in ["train", "val"]:
        losses = []

        for _ in range(steps):
            x, y = TinyShakespeare.get_random_batch(
                split, sequence_length=sequence_length
            )

            logits = model(x)

            batch_size, sequence_length, vocab_size = logits.shape
            logits = logits.view(batch_size * sequence_length, vocab_size)
            y = y.view(-1)

            loss = F.cross_entropy(logits, y)
            losses.append(loss.item())

        out[split] = sum(losses) / len(losses)

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=int, default=3e-4)
    parser.add_argument("--steps", type=int, default=4500)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--embed-size", type=int, default=384)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--num-blocks", type=int, default=6)
    args = parser.parse_args()

    model = Transfomer(
        len(TinyShakespeare._vocab),
        args.sequence_length,
        args.embed_size,
        args.num_heads,
        args.num_blocks,
        dropout=0.2,
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    for i in range(args.steps):
        if i % 300 == 0:
            model.eval()
            losses = estimate_loss(model, args.sequence_length)
            print(f"Step {i}: {losses}")
            model.train()

        x, y = TinyShakespeare.get_random_batch(
            "train", args.batch_size, args.sequence_length
        )

        logits = model(x)

        # TODO: Meh, I'm not really too convinced about this reshaping
        batch_size, sequence_length, vocab_size = logits.shape
        logits = logits.view(batch_size * sequence_length, vocab_size)
        y = y.view(-1)

        loss = F.cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    tokens = model.yap(tensor([[0]]), 1000)
    for token in tokens[0]:
        print(TinyShakespeare._token_to_char[token.item()], end="")
    print()
