import torch
import torch.nn.functional as F
from torch.optim import Adam
from dataset import get_flickr8k_loader
from model import TransformerCaptionModel
import os


def train():
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    loader, dataset = get_flickr8k_loader("data/flickr8k", batch_size=8)

    model = TransformerCaptionModel(
        vocab_size=len(dataset.word2idx)
    ).to(device)

    optimizer = Adam(model.parameters(), lr=1e-4)
    pad_idx = dataset.word2idx["<PAD>"]

    model.train()
    for epoch in range(20):
        total_loss = 0.0

        for feats,boxes, caps in loader:
            feats = feats.to(device)
            caps = caps.to(device)
            boxes = boxes.to(device)

            optimizer.zero_grad()

            inp = caps[:, :-1]
            tgt = caps[:, 1:]

            logits = model(feats,boxes, inp)

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tgt.reshape(-1),
                ignore_index=pad_idx,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
        if torch.isnan(loss):
          print("NaN detected. Stopping training.")
          return

        print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")

    os.makedirs("experiments", exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "word2idx": dataset.word2idx,
            "idx2word": dataset.idx2word,
        },
        "experiments/flickr8k_baseline.pt",
    )

    print("Training complete. Model saved.")


if __name__ == "__main__":
    train()
