from tinygrad import Tensor, nn

from bigram import Bigram
from dataset import TinyShakespeare

train_length = len(TinyShakespeare._train_tokens)
val_length = len(TinyShakespeare._val_tokens)
total_length = train_length + val_length

print(f"Train: {train_length} tokens ({train_length / total_length:.1%})")
print(f"Val: {val_length} tokens ({val_length / total_length:.1%})")

x, y = TinyShakespeare.get_random_batch("train")

print("x:", x.numpy())
print("y:", y.numpy())

model = Bigram(vocab_size=len(TinyShakespeare._vocab))
text = model.yap(0, length=100)

for token in text:
    print(TinyShakespeare._token_to_char[token], end="")

optimizer = nn.optim.AdamW([model.embedding.weight], lr=0.001)

with Tensor.train():
    for steps in range(1000):
        x, y = TinyShakespeare.get_random_batch("train", batch_size=32)

        out = model(x)
        batch, sequence_length, vocab_size = out.shape
        logits = out.view(batch * sequence_length, len(TinyShakespeare._vocab))

        target = y.view(batch * sequence_length, 1)
        loss = logits.sparse_categorical_crossentropy(target) / target.shape[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if steps % 100 == 0:
            print(f"step {steps}, loss: {loss.numpy()}")

text = model.yap(0, length=100)

for token in text:
    print(TinyShakespeare._token_to_char[token], end="")
