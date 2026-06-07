"""
Convert Amazon Fine Food Reviews from SNAP text format to CSV.

SNAP format (one block per review, blank-line separated):
    product/productId: ...
    review/score: 5.0
    review/text: ...

Output: Reviews.csv with columns [Id, Score, Text]
"""
import csv
import gzip
import sys
from pathlib import Path


def parse_snap(gz_path: str):
    """Yield dicts for each review block."""
    block = {}
    with gzip.open(gz_path, "rt", encoding="latin-1") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if line == "":
                if block:
                    yield block
                    block = {}
                continue
            if ": " in line:
                key, _, value = line.partition(": ")
                block[key.strip()] = value.strip()
        if block:
            yield block


def main(gz_path: str, out_path: str) -> None:
    print(f"Converting {gz_path} -> {out_path} ...")
    written = 0
    with open(out_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["Id", "Score", "Text"])
        writer.writeheader()
        for i, block in enumerate(parse_snap(gz_path)):
            score = block.get("review/score", "")
            text = block.get("review/text", "")
            if score and text:
                writer.writerow({"Id": i + 1, "Score": score, "Text": text})
                written += 1
    print(f"Done — {written:,} reviews written to {out_path}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    src = str(root / "data" / "raw" / "finefoods.txt.gz")
    dst = str(root / "data" / "raw" / "Reviews.csv")
    main(src, dst)
