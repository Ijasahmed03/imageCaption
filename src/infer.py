# src/infer.py

import torch
from dataset import get_dataloader
from model import TransformerCaptionModel


def decode(tokens, idx2word):
    words = []
    for t in tokens:
        word = idx2word.get(int(t), "<UNK>")
        if word == "<EOS>":
            break
        if word not in ("<SOS>", "<PAD>"):
            words.append(word)
    return " ".join(words)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    loader, dataset = get_dataloader("data/sample", batch_size=2, shuffle=False)

    model = TransformerCaptionModel(
        vocab_size=len(dataset.word2idx)
    ).to(device)

    # ⚠️ IMPORTANT:
    # For now, we reuse the freshly trained model weights in memory.
    # Later we will load from a saved checkpoint.

    model.eval()

    features, captions = next(iter(loader))
    features = features.to(device)

    sos = dataset.word2idx["<SOS>"]
    eos = dataset.word2idx["<EOS>"]

    generated = model.generate(features, sos, eos)

    print("\nGenerated captions:\n")
    for i in range(generated.size(0)):
        sentence = decode(generated[i].cpu().tolist(), dataset.idx2word)
        print(f"{i+1}. {sentence}")


if __name__ == "__main__":
    main()
