import streamlit as st
import torch
from pathlib import Path
from generate import generate_poem, load_model


# Config
st.set_page_config(page_title="Poem Generator by zeyneppinarsoy", layout="centered")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATHS = {
    "Emily Dickinson": "models/dickinson_model.pt",
    "Walt Whitman": "models/whitman_model.pt",
    "William Shakespeare": "models/shakespeare_model.pt"
}


# UI
st.title("📝 Poem Generator by zeyneppinarsoy")
st.write("Char-level LSTM poem generator")

author = st.selectbox("Pick someone", list(MODEL_PATHS.keys()))

seed = st.text_area(
    "Seed (opening text)",
    value="<POEM_START>\n",
    height=120
)

temperature = st.slider(
    "Temperature (creativity)",
    min_value=0.2,
    max_value=1.2,
    value=0.85,
    step=0.05
)

uploaded_file = st.file_uploader(
    "Load TXT File (optional)",
    type=["txt"]
)

max_chars = st.number_input(
    "Max character",
    min_value=100,
    max_value=1500,
    value=500,
    step=50
)


# Load uploaded seed
if uploaded_file is not None:
    seed = uploaded_file.read().decode("utf-8")


# Generate
if st.button("✨ Generate Poem"):
    model_path = MODEL_PATHS[author]

    if not Path(model_path).exists():
        st.error(f"Model could not found: {model_path}")
    else:
        with st.spinner("Loading model right now..."):
            model, char2idx, idx2char = load_model(
                model_path,
                device=DEVICE
            )

        with st.spinner("Generating poem..."):
            poem = generate_poem(
                model=model,
                char2idx=char2idx,
                idx2char=idx2char,
                seed=seed,
                temperature=temperature,
                max_chars=max_chars,
                device=DEVICE
            )

        st.subheader("📜 Generated Poem")
        st.text(poem)
