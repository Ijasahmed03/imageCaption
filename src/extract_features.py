# src/extract_features.py

import os
import torch
import torchvision
import numpy as np
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_REGIONS = 36

# Image preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load pretrained Faster R-CNN
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
model.to(DEVICE)
model.eval()


def extract_from_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).to(DEVICE)

    with torch.no_grad():
        outputs = model([img_tensor])[0]

    boxes = outputs["boxes"]
    num = min(len(boxes), NUM_REGIONS)

    # Initialize fixed-size arrays
    appearance = np.zeros((NUM_REGIONS, 2048), dtype="float32")
    geom = np.zeros((NUM_REGIONS, 4), dtype="float32")

    if num > 0:
        selected_boxes = boxes[:num].cpu().numpy()

        # Normalize box coordinates
        w, h = img.size
        selected_boxes[:, 0] /= w
        selected_boxes[:, 1] /= h
        selected_boxes[:, 2] /= w
        selected_boxes[:, 3] /= h

        geom[:num] = selected_boxes

        # Placeholder appearance features
        appearance[:num] = np.random.randn(num, 2048).astype("float32")

    return appearance, geom



def main():
    image_dir = "data/flickr8k/Images"
    save_dir = "data/flickr8k/features"
    os.makedirs(save_dir, exist_ok=True)

    features_all = []
    boxes_all = []

    image_files = sorted(os.listdir(image_dir))

    for img_name in tqdm(image_files):
        img_path = os.path.join(image_dir, img_name)
        feats, boxes = extract_from_image(img_path)

        features_all.append(feats)
        boxes_all.append(boxes)

    np.save(os.path.join(save_dir, "features.npy"), np.array(features_all))
    np.save(os.path.join(save_dir, "boxes.npy"), np.array(boxes_all))

    print("Feature extraction completed.")
    print("Features shape:", np.array(features_all).shape)
    print("Boxes shape:", np.array(boxes_all).shape)


if __name__ == "__main__":
    main()
