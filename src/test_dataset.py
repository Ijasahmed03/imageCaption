from dataset import get_dataloader

loader, dataset = get_dataloader("data/sample")

for feats, caps in loader:
    print("features shape:", feats.shape)
    print("captions shape:", caps.shape)
    break
