
print(">>> infer_flickr8k.py file loaded")
import torch
from dataset import get_flickr8k_loader
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
    print("Starting inference script...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    print("Loading checkpoint...")
    checkpoint = torch.load(
        "experiments/flickr8k_baseline.pt", map_location=device
    )

    print("Initializing dataset & dataloader...")
    loader, dataset = get_flickr8k_loader(
        "data/flickr8k", batch_size=4, shuffle=False
    )

    print("Building model...")
    model = TransformerCaptionModel(
        vocab_size=len(checkpoint["word2idx"])
    ).to(device)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dataset.word2idx = checkpoint["word2idx"]
    dataset.idx2word = checkpoint["idx2word"]

    print("Fetching first batch (this may take a bit)...")
    feats, boxes, _ = next(iter(loader))
    print("Batch loaded.")

    feats = feats.to(device)
    boxes = boxes.to(device)
    sos = dataset.word2idx["<SOS>"]
    eos = dataset.word2idx["<EOS>"]

    print("Generating captions...")
    generated = model.generate(feats,boxes, sos, eos)

    print("\nGenerated captions:\n")
    for i in range(generated.size(0)):
        caption = decode(generated[i].cpu().tolist(), dataset.idx2word)
        print(f"{i+1}. {caption}")
    print("\nInference complete.")  

if __name__ == "__main__":
    main()
