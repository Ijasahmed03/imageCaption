input_file = "data/flickr8k/captions.txt"
output_file = "data/flickr8k/captions_fixed.txt"

with open(input_file, "r", encoding="utf-8") as fin, \
        open(output_file, "w", encoding="utf-8") as fout:

    header = fin.readline()  # skip header line

    for line in fin:
        line = line.strip()
        if not line:
            continue

        parts = line.split("|")
        if len(parts) != 3:
            continue

        img = parts[0]
        caption = parts[2]

        fout.write(f"{img},{caption}\n")

print("✅ captions_fixed.txt created successfully")
