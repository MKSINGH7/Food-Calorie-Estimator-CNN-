"""Build the train/validation folders.

Option A - Food-101 (downloads ~4.6 GB once, extracts only the classes in data/calories.json):
    python -m src.prepare_data
    python -m src.prepare_data --train-per-class 400 --val-per-class 100

Option B - your own photos, laid out as  my_photos/<class_name>/*.jpg :
    python -m src.prepare_data --source-dir my_photos

Option C - Food-101 already downloaded & extracted somewhere:
    python -m src.prepare_data --food101-dir /path/to/food-101

Result:
    dataset/train/<class>/*.jpg
    dataset/validation/<class>/*.jpg
"""
import argparse
import random
import shutil
import sys
import tarfile
from pathlib import Path

import requests

from . import calories, config

FOOD101_URL = "https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def download(url: str, dest: Path):
    if dest.exists():
        print(f"Using existing archive {dest}")
        return
    print(f"Downloading {url}\n  -> {dest}  (about 4.6 GB, this takes a while)")
    tmp = dest.with_suffix(".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if done % (100 << 20) < (1 << 20):
                    pct = f"{done / total:.0%}" if total else ""
                    print(f"  {done >> 20} MB {pct}", flush=True)
    tmp.rename(dest)


def extract_classes(tar_path: Path, classes, staging: Path):
    """Single sequential pass through the tarball, keeping only the wanted classes."""
    wanted = {f"food-101/images/{c}/" for c in classes}
    print(f"Extracting {len(classes)} classes from {tar_path} ...")
    n = 0
    with tarfile.open(tar_path, "r:gz") as tar:
        for m in tar:
            if not m.isfile() or not m.name.endswith(".jpg"):
                continue
            prefix = m.name.rsplit("/", 1)[0] + "/"
            if prefix not in wanted:
                continue
            cls = prefix.split("/")[-2]
            out = staging / cls
            out.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(m) as src, open(out / Path(m.name).name, "wb") as dst:
                shutil.copyfileobj(src, dst)
            n += 1
            if n % 2000 == 0:
                print(f"  {n} images", flush=True)
    print(f"Extracted {n} images")


def split(src_root: Path, classes, out_root: Path, train_n, val_n, move, seed=42):
    for sub in ("train", "validation"):
        if (out_root / sub).exists():
            shutil.rmtree(out_root / sub)
    rng = random.Random(seed)
    for cls in classes:
        files = sorted(p for p in (src_root / cls).glob("*") if p.suffix.lower() in IMG_EXT)
        if len(files) < 10:
            sys.exit(f"Class '{cls}' has only {len(files)} images in {src_root / cls}")
        rng.shuffle(files)
        n_val = min(val_n, max(1, int(len(files) * 0.2)))
        val_files = files[:n_val]
        train_files = files[n_val : n_val + train_n]
        for sub, group in (("validation", val_files), ("train", train_files)):
            d = out_root / sub / cls
            d.mkdir(parents=True, exist_ok=True)
            for f in group:
                (shutil.move if move else shutil.copy2)(str(f), d / f.name)
        print(f"  {cls:22s} train={len(train_files):4d} val={len(val_files):4d}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--out", default=str(config.DATA_DIR))
    p.add_argument("--source-dir", help="your own photos: <dir>/<class>/*.jpg")
    p.add_argument("--food101-dir", help="existing extracted food-101 folder (contains images/)")
    p.add_argument("--classes", nargs="*", help="class names (default: all in data/calories.json)")
    p.add_argument("--train-per-class", type=int, default=600)
    p.add_argument("--val-per-class", type=int, default=150)
    args = p.parse_args()

    out = Path(args.out)
    classes = sorted(args.classes) if args.classes else calories.food_classes()

    if args.source_dir:
        src = Path(args.source_dir)
        found = sorted(d.name for d in src.iterdir() if d.is_dir())
        classes = [c for c in classes if c in found] if args.classes else found
        print(f"Using your photos from {src}: {len(classes)} classes")
        split(src, classes, out, args.train_per_class, args.val_per_class, move=False)
        print("\nNote: add any new class to data/calories.json so calories can be looked up.")
    elif args.food101_dir:
        split(Path(args.food101_dir) / "images", classes, out, args.train_per_class, args.val_per_class, move=False)
    else:
        archive = config.ROOT / "food-101.tar.gz"
        staging = config.ROOT / "_staging"
        download(FOOD101_URL, archive)
        extract_classes(archive, classes, staging)
        split(staging, classes, out, args.train_per_class, args.val_per_class, move=True)
        shutil.rmtree(staging, ignore_errors=True)
        print("\nYou can delete food-101.tar.gz now to free ~4.6 GB.")
    print(f"\nDone. Dataset in {out}. Next: python -m src.train")


if __name__ == "__main__":
    main()
