from dataset import get_flickr8k_loader

loader, _ = get_flickr8k_loader("data/flickr8k", batch_size=4)
feats, boxes, caps = next(iter(loader))

print(feats.shape)
print(boxes.shape)
print(caps.shape)
print("Data loading test complete.")