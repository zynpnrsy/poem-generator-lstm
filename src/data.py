import torch
from torch.utils.data import Dataset
from pathlib import Path


class PoemDataset(Dataset):
    """
    Character-level dataset for poem style learning.
    Generates (input_seq, target_seq) pairs where:
    input  = [c1, c2, c3, ...]
    target = [c2, c3, c4, ...]
    """

    def __init__(self, text, seq_length=50):
        self.seq_length = seq_length

        # Build vocab
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)

        self.char2idx = {c: i for i, c in enumerate(self.chars)}
        self.idx2char = {i: c for c, i in self.char2idx.items()}

        # Encode full text
        encoded = torch.tensor(
            [self.char2idx[c] for c in text],
            dtype=torch.long
        )

        # Create sequence to sequence pairs
        self.inputs = []
        self.targets = []

        for i in range(len(encoded) - seq_length):
            self.inputs.append(encoded[i:i + seq_length])
            self.targets.append(encoded[i + 1:i + seq_length + 1])

        self.inputs = torch.stack(self.inputs)
        self.targets = torch.stack(self.targets)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

    def save_mappings(self, path="char_mappings.pth"):
        """
        Save character mappings for generation phase.
        """
        torch.save({
            "char2idx": self.char2idx,
            "idx2char": self.idx2char,
            "vocab_size": self.vocab_size
        }, path)


def load_text(file_path):
    """
    Load poem text from file.
    Keeps POEM_START and POEM_END tokens.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(path, "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    # Quick sanity check
    text = load_text("data/poem_list.txt")
    dataset = PoemDataset(text, seq_length=50)

    print(f"Total sequences: {len(dataset)}")
    print(f"Vocabulary size: {dataset.vocab_size}")

    x, y = dataset[0]
    print("Input example:", x[:10])
    print("Target example:", y[:10])

    dataset.save_mappings()
    print("char_mappings.pth saved")
