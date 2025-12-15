"""
Extracts a fixed size subset of Amazon Electronics reviews to create a manageable dataset
for development and experimentation.
extracts the first 50,000 review records.

"""

import gzip
import json

input_path = "data/raw/electronics_full.jsonl.gz"
output_path = "data/raw/electronics_50k.jsonl"

limit = 50000
count = 0

with gzip.open(input_path, "rt", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
    for line in infile:
        outfile.write(line)
        count += 1

        if count == limit:
            break

print(f"Saved first {limit} reviews to {output_path}")
