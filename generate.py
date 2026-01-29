import torch
from src.model import CharLSTM

# Load model
def load_model(model_path, device="cpu"):
    checkpoint = torch.load(model_path, map_location=device)

    model_args = checkpoint["model_args"]

    model = CharLSTM(
        vocab_size=checkpoint["vocab_size"],
        embed_dim=model_args["embed_dim"],
        hidden_dim=model_args["hidden_dim"],
        num_layers=model_args["num_layers"],
        dropout=0.0
    )

    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint["char2idx"], checkpoint["idx2char"]


# Generate poem

@torch.no_grad()
def generate_poem(
    model,
    char2idx,
    idx2char,
    seed="<POEM_START>\n",
    max_chars=500,
    temperature=0.85,
    device="cpu"
):
    model.eval()

    seed_indices = [char2idx[c] for c in seed if c in char2idx]
    if len(seed_indices) == 0:
        seed_indices = [char2idx["<"]]  # fallback

    hidden = model.init_hidden(1, device)

    x = torch.tensor([seed_indices], dtype=torch.long).to(device)
    _, hidden = model(x, hidden)

    current_char = seed_indices[-1]
    generated = seed_indices.copy()

    for _ in range(max_chars):
        x = torch.tensor([[current_char]], dtype=torch.long).to(device)
        logits, hidden = model(x, hidden)

        logits = logits[0, -1] / temperature
        probs = torch.softmax(logits, dim=0)

        next_char = torch.multinomial(probs, 1).item()
        generated.append(next_char)
        current_char = next_char

        if "".join(idx2char[i] for i in generated[-12:]) == "<POEM_END>":
            break

    return "".join(idx2char[i] for i in generated)
