"""Calorie lookup table (data/calories.json)."""
import json
from functools import lru_cache

from .config import CALORIES_PATH


@lru_cache(maxsize=1)
def load_table() -> dict:
    with open(CALORIES_PATH, encoding="utf-8") as f:
        table = json.load(f)
    return {k: v for k, v in table.items() if not k.startswith("_")}


def food_classes() -> list:
    """Sorted class names (same order Keras uses when reading class folders)."""
    return sorted(load_table().keys())


def lookup(food: str, grams: float | None = None) -> dict:
    """Return calories for `food`. If grams is None the typical serving is used."""
    entry = load_table().get(food)
    if entry is None:
        return {"kcal_per_100g": None, "grams": grams, "total_kcal": None, "serving": None}
    grams = float(grams) if grams else float(entry["serving_g"])
    return {
        "kcal_per_100g": entry["kcal_per_100g"],
        "grams": round(grams, 1),
        "total_kcal": round(entry["kcal_per_100g"] * grams / 100.0),
        "serving": entry.get("serving"),
    }
