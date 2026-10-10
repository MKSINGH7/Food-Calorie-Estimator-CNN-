"""Create a tiny synthetic image dataset so you can smoke-test the whole pipeline
in a minute without downloading Food-101.   (Accuracy numbers are meaningless.)

    python scripts/make_dummy_dataset.py --out dataset_dummy --classes 4
"""
import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent


def make(cls_idx, rng, size=128):
    base = [(200, 60, 60), (60, 160, 70), (70, 90, 200), (220, 190, 50), (160, 80, 180), (60, 190, 190)][cls_idx % 6]
    img = Image.new("RGB", (size, size), tuple(max(0, min(255, c + rng.randint(-25, 25))) for c in base))
    d = ImageDraw.Draw(img)
    for _ in range(6):
        x, y = rng.randint(0, size), rng.randint(0, size)
        r = rng.randint(8, 30)
        col = tuple(255 - c for c in base)
        (d.ellipse if cls_idx % 2 else d.rectangle)([x, y, x + r, y + r], fill=col)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "dataset_dummy"))
    ap.add_argument("--classes", type=int, default=4)
    ap.add_argument("--train", type=int, default=40)
    ap.add_argument("--val", type=int, default=12)
    a = ap.parse_args()

    names = sorted(k for k in json.loads((ROOT / "data" / "calories.json").read_text()) if not k.startswith("_"))[: a.classes]
    rng = random.Random(0)
    for split, n in (("train", a.train), ("validation", a.val)):
        for i, name in enumerate(names):
            d = Path(a.out) / split / name
            d.mkdir(parents=True, exist_ok=True)
            for j in range(n):
                make(i, rng).save(d / f"{j:03d}.jpg")
    print(f"Dummy dataset with classes {names} -> {a.out}")


if __name__ == "__main__":
    main()
