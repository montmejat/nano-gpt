import numpy as np
from tinygrad import Tensor, nn


class Bigram:
    def __init__(self, vocab_size: int):
        self.embedding = nn.Embedding(vocab_size=vocab_size, embed_size=vocab_size)

    def __call__(self, x: Tensor):
        return self.embedding(x)

    @property
    def weight(self):
        return self.embedding.weight

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

    def yap(self, first_token: int, length: int):
        tokens = Tensor([[first_token]])

        for _ in range(length):
            x = tokens[:, -self.sequence_length :]
            out = self(x)
            out = out[:, -1, :].squeeze()  # Get last token
            out = out.softmax()  # Get probabilities
            # Sample a random token from the distribution
            token = np.random.multinomial(1, out.numpy()).argmax()
            tokens = tokens.cat(Tensor([[token]]), dim=1)

        return tokens
