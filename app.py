from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
import numpy as np

from src.model import TransformerCaptionModel
from src.dataset import Flickr8kDataset


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow all origins (OK for development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------- Load model ONCE --------
checkpoint = torch.load(
    "experiments/flickr8k_baseline.pt", map_location=DEVICE)

model = TransformerCaptionModel(
    vocab_size=len(checkpoint["word2idx"])
).to(DEVICE)
model.load_state_dict(checkpoint["model_state"])
model.eval()

word2idx = checkpoint["word2idx"]
idx2word = checkpoint["idx2word"]

sos = word2idx["<SOS>"]
eos = word2idx["<EOS>"]

# Dummy feature extractor (same logic as training)


def extract_dummy_features():
    feats = np.random.randn(1, 36, 2048).astype("float32")
    boxes = np.random.rand(1, 36, 4).astype("float32")
    return (
        torch.tensor(feats).to(DEVICE),
        torch.tensor(boxes).to(DEVICE),
    )


def decode(tokens):
    words = []
    for t in tokens:
        w = idx2word.get(int(t), "<UNK>")

        if w == "<EOS>":
            break
        if w in ("<SOS>", "<PAD>"):
            continue

        # UI-friendly replacement
        if w == "<UNK>":
            w = "something"

        words.append(w)

    return " ".join(words) if words else "no caption generated"


@app.post("/caption")
async def caption_image(file: UploadFile = File(...)):
    try:
        _ = Image.open(file.file).convert("RGB")

        feats, boxes = extract_dummy_features()

        with torch.no_grad():
            output = model.generate(feats, boxes, sos, eos)

        tokens = output[0].cpu().tolist()

        print("DEBUG raw tokens:", tokens)
        print("DEBUG decoded tokens:", [
              idx2word.get(t, "<?>") for t in tokens])

        caption = decode(tokens)

        return {
            "tokens": tokens,
            "caption": caption
        }

    except Exception as e:
        print("🔥 ERROR:", e)
        return {"error": str(e)}
