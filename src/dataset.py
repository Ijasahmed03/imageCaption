# src/dataset.py

import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import nltk

# make sure punkt is available
#nltk.download("punkt", quiet=True)


class CaptionDataset(Dataset):
    def __init__(self, data_dir, max_len=20):
        self.data_dir = data_dir
        self.max_len = max_len

        # load features
        feat_path = os.path.join(data_dir, "features.npy")
        self.features = np.load(feat_path)  # (num_images, num_regions, feat_dim)

        # load captions
        cap_path = os.path.join(data_dir, "captions.json")
        with open(cap_path, "r") as f:
            data = json.load(f)

        self.image_ids = data["image_ids"]
        self.captions = data["captions"]

        # build samples (1 caption per image for now)
        self.samples = []
        for img_id in self.image_ids:
            cap = self.captions[str(img_id)][0]
            self.samples.append((img_id, cap))

        # vocabulary
        self.pad = "<PAD>"
        self.sos = "<SOS>"
        self.eos = "<EOS>"
        self.unk = "<UNK>"

        self.word2idx = {
            self.pad: 0,
            self.sos: 1,
            self.eos: 2,
            self.unk: 3
        }

        self.idx2word = {}

        self._build_vocab()

    def _build_vocab(self):
        counter = Counter()

        for _, caption in self.samples:
            tokens = nltk.word_tokenize(caption.lower())
            counter.update(tokens)

        idx = len(self.word2idx)
        for word in counter:
            self.word2idx[word] = idx
            idx += 1

        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def encode_caption(self, caption):
        tokens = nltk.word_tokenize(caption.lower())
        ids = [self.word2idx.get(t, self.word2idx[self.unk]) for t in tokens]

        ids = [self.word2idx[self.sos]] + ids + [self.word2idx[self.eos]]

        # pad or trim
        if len(ids) < self.max_len:
            ids += [self.word2idx[self.pad]] * (self.max_len - len(ids))
        else:
            ids = ids[:self.max_len]

        return torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_id, caption = self.samples[idx]

        feats = self.features[img_id]
        feats = torch.tensor(feats, dtype=torch.float32)

        cap_ids = self.encode_caption(caption)

        return feats, cap_ids


def get_dataloader(data_dir, batch_size=4, shuffle=True):
    dataset = CaptionDataset(data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader, dataset

class Flickr8kDataset(Dataset):
    def __init__(self, data_dir, max_len=20):
        self.max_len = max_len

        # Load features
        self.features = np.load(os.path.join(data_dir, "features", "features.npy"))
        #load boxes
        self.boxes = np.load(os.path.join(data_dir, "features", "boxes.npy"))

        # Load captions
        caption_file = os.path.join(data_dir, "captions.txt")
        self.image_captions = {}

        with open(caption_file, "r", encoding="utf-8") as f:
            for line in f:
                img, caption = line.strip().split(",", 1)
                img_id = img.split(".")[0]
                self.image_captions.setdefault(img_id, []).append(caption)

        with open(os.path.join(data_dir, "features", "image_ids.txt")) as f:
            self.image_ids = [line.strip() for line in f]


        # Special tokens
        self.pad = "<PAD>"
        self.sos = "<SOS>"
        self.eos = "<EOS>"
        self.unk = "<UNK>"

        self.word2idx = {
            self.pad: 0,
            self.sos: 1,
            self.eos: 2,
            self.unk: 3,
        }

        self._build_vocab()

    def _build_vocab(self):
        counter = Counter()
        for caps in self.image_captions.values():
            for cap in caps:
                tokens = nltk.word_tokenize(cap.lower())
                counter.update(tokens)

        idx = len(self.word2idx)
        for word in counter:
            self.word2idx[word] = idx
            idx += 1

        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def encode_caption(self, caption):
        tokens = nltk.word_tokenize(caption.lower())
        ids = [self.word2idx.get(t, self.word2idx[self.unk]) for t in tokens]
        ids = [self.word2idx[self.sos]] + ids + [self.word2idx[self.eos]]

        if len(ids) < self.max_len:
            ids += [self.word2idx[self.pad]] * (self.max_len - len(ids))
        else:
            ids = ids[:self.max_len]

        return torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        box = torch.tensor(self.boxes[idx], dtype=torch.float32)
        caption = np.random.choice(self.image_captions[img_id])
        cap_ids = self.encode_caption(caption)
        
        return feature,box, cap_ids


def get_flickr8k_loader(data_dir, batch_size=16, shuffle=True):
    dataset = Flickr8kDataset(data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader, dataset
