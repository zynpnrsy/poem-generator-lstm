import torch
import torch.nn as nn


class CharLSTM(nn.Module):
    """
    Character-level LSTM for poetry style modeling.
    Trained to learn character distributions of a specific author 
    """

    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.2
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Character embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # LSTM backbone
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Output projection
        self.fc = nn.Linear(hidden_dim, vocab_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden=None):
        """
        x: (batch_size, seq_len)
        """
        x = self.embedding(x)
        x = self.dropout(x)

        out, hidden = self.lstm(x, hidden)
        out = self.dropout(out)

        logits = self.fc(out)  # (batch, seq_len, vocab_size)
        return logits, hidden

    def init_hidden(self, batch_size, device):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return h0, c0

    @torch.no_grad()
    def predict_next_char(self, char_idx, hidden, temperature=1.0):
        """
        Used ONLY during generation (not training)
        """
        x = torch.tensor([[char_idx]], dtype=torch.long).to(next(self.parameters()).device)
        logits, hidden = self.forward(x, hidden)

        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)

        next_char = torch.multinomial(probs, num_samples=1).item()
        return next_char, hidden


if __name__ == "__main__":
    # quick sanity check
    vocab_size = 100
    model = CharLSTM(vocab_size)

    x = torch.randint(0, vocab_size, (4, 50))
    out, hidden = model(x)

    print("Output shape:", out.shape)
