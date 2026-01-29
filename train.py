import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model import CharLSTM
from src.data import PoemDataset, load_text

# Args
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, required=True)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--epochs", type=int, default=25)
parser.add_argument("--batch_size", type=int, default=32)
args = parser.parse_args()

# Config
SEQ_LENGTH = 50
EMBED_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.2
LR = 0.0005

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)


# Dataset
text = load_text(args.data)
dataset = PoemDataset(text, seq_length=SEQ_LENGTH)

dataloader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True
)

print("Sequences:", len(dataset))
print("Vocab size:", dataset.vocab_size)


# Model
model = CharLSTM(
    vocab_size=dataset.vocab_size,
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Training
for epoch in range(args.epochs):
    model.train()
    total_loss = 0
    hidden = model.init_hidden(args.batch_size, DEVICE)

    for x, y in dataloader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        hidden = tuple(h.detach() for h in hidden)

        optimizer.zero_grad()
        outputs, hidden = model(x, hidden)

        loss = criterion(
            outputs.view(-1, dataset.vocab_size),
            y.view(-1)
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{args.epochs} | Loss: {total_loss / len(dataloader):.4f}")


# Save (MODEL + VOCAB TOGETHER)
torch.save({
    "state_dict": model.state_dict(),
    "char2idx": dataset.char2idx,
    "idx2char": dataset.idx2char,
    "vocab_size": dataset.vocab_size,
    "model_args": {
        "embed_dim": EMBED_DIM,
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS
    }
}, args.out)

print(f"Model saved to {args.out}")
