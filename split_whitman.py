from pathlib import Path

# =============================
# AYARLAR
# =============================
INPUT_FILE = "data/whitman_poems_pre.txt"
OUTPUT_FILE = "data/whitman_poems.txt"

MIN_CHARS = 200
MAX_CHARS = 400

POEM_START = "<POEM_START>\n"
POEM_END = "\n<POEM_END>\n\n"


def split_whitman_poems(text):
    poems = []
    current_chunk = ""

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        # Satırı ekle
        if len(current_chunk) + len(line) < MAX_CHARS:
            current_chunk += line + "\n"
        else:
            # Chunk yeterince büyükse kaydet
            if len(current_chunk) >= MIN_CHARS:
                poems.append(current_chunk.strip())
            current_chunk = line + "\n"

    # Son parça
    if len(current_chunk) >= MIN_CHARS:
        poems.append(current_chunk.strip())

    return poems


def main():
    raw_text = Path(INPUT_FILE).read_text(encoding="utf-8")

    poem_chunks = split_whitman_poems(raw_text)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for poem in poem_chunks:
            f.write(POEM_START)
            f.write(poem)
            f.write(POEM_END)

    print(f"Toplam parça sayısı: {len(poem_chunks)}")
    print(f"Dataset yazıldı: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
