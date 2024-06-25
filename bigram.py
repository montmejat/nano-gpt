import numpy as np
from tinygrad import Tensor, nn


class Bigram:
    def __init__(self, vocab_size: int):
        self.embedding = nn.Embedding(vocab_size=vocab_size, embed_size=vocab_size)

    def __call__(self, x: Tensor):
        return self.embedding(x)

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
