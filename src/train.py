# src/train.py

import torch
import torch.nn.functional as F
from torch.optim import Adam
from dataset import get_dataloader
from model import TransformerCaptionModel


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    loader, dataset = get_dataloader("data/sample", batch_size=4)

    model = TransformerCaptionModel(
        vocab_size=len(dataset.word2idx)
    ).to(device)

    optimizer = Adam(model.parameters(), lr=1e-4)

    pad_idx = dataset.word2idx["<PAD>"]

    model.train()
    for epoch in range(3):
        total_loss = 0

        for features, captions in loader:
            features = features.to(device)
            captions = captions.to(device)

            optimizer.zero_grad()

            # input captions (remove last token)
            input_caps = captions[:, :-1]
            target_caps = captions[:, 1:]

            logits = model(features, input_caps)

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_caps.reshape(-1),
                ignore_index=pad_idx,
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")


if __name__ == "__main__":
    train()
