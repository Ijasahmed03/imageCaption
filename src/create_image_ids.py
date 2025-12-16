import os

image_dir = "data/flickr8k/Images"
out_file = "data/flickr8k/features/image_ids.txt"

image_files = sorted(os.listdir(image_dir))

with open(out_file, "w") as f:
    for name in image_files:
        f.write(name.split(".")[0] + "\n")

print("image_ids.txt created")
