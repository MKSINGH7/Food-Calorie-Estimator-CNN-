"""Inference: CNN prediction + calorie lookup (+ optional LLM).

CLI:
    python -m src.predict path/to/photo.jpg
    python -m src.predict photo.jpg --grams 250
    python -m src.predict photo.jpg --llm anthropic        # nutrition advice + low-confidence fallback
    python -m src.predict photo.jpg --llm ollama
    python -m src.predict photo.jpg --llm anthropic --llm-only   # no CNN needed at all
"""
import argparse
import json

import numpy as np
from PIL import Image

from . import calories, config, llm


class Predictor:
    """Loads the trained Keras model once; thread-safe enough for a small API."""

    def __init__(self, model_path=config.MODEL_PATH, meta_path=config.META_PATH):
        from tensorflow import keras

        if not model_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"No trained model at {model_path}. Run `python -m src.train` first "
                "(or use --llm-only with an LLM provider)."
            )
        self.meta = json.loads(meta_path.read_text())
        self.class_names = self.meta["class_names"]
        self.img_size = int(self.meta["img_size"])
        self.model = keras.models.load_model(model_path)

    def top_k(self, image: Image.Image, k=3):
        img = image.convert("RGB").resize((self.img_size, self.img_size))
        x = np.asarray(img, dtype="float32")[None, ...]  # raw 0-255; model rescales itself
        probs = self.model.predict(x, verbose=0)[0]
        idx = np.argsort(probs)[::-1][:k]
        return [{"food": self.class_names[i], "confidence": round(float(probs[i]), 4)} for i in idx]


def analyze(image, predictor=None, grams=None, provider=None, use_llm=False, llm_only=False, top_k=3):
    """Full pipeline. Returns a JSON-serialisable dict.

    use_llm  -> add nutrition advice, and fall back to LLM vision if the CNN is not confident.
    llm_only -> skip the CNN and let the LLM identify the food from the image.
    """
    provider = provider or config.LLM_PROVIDER
    notes = []

    if llm_only or predictor is None:
        if not llm.is_enabled(provider):
            raise llm.LLMError("LLM-only mode needs LLM_PROVIDER=anthropic|ollama")
        est = llm.estimate_from_image(image, provider)
        est.update(source=f"llm:{provider}", top_k=[], serving=None)
        return est

    preds = predictor.top_k(image, top_k)
    best = preds[0]

    # Low-confidence -> let the LLM look at the picture.
    if use_llm and llm.is_enabled(provider) and best["confidence"] < config.CONF_THRESHOLD:
        try:
            est = llm.estimate_from_image(image, provider)
            est.update(source=f"llm:{provider}", top_k=preds, serving=None)
            est["notes"] = (
                f"CNN was unsure ({best['food']} {best['confidence']:.0%}); LLM estimate used. "
                + str(est.get("notes", ""))
            )
            return est
        except llm.LLMError as e:
            notes.append(f"LLM fallback failed, using CNN result: {e}")

    cal = calories.lookup(best["food"], grams)
    result = {
        "source": "cnn",
        "food": best["food"],
        "confidence": best["confidence"],
        "top_k": preds,
        **cal,
    }
    if use_llm and llm.is_enabled(provider) and cal["total_kcal"] is not None:
        try:
            result["advice"] = llm.advice(best["food"], cal["kcal_per_100g"], cal["grams"], cal["total_kcal"], provider)
        except llm.LLMError as e:
            notes.append(f"LLM advice failed: {e}")
    if notes:
        result["notes"] = " ".join(notes)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("image")
    p.add_argument("--grams", type=float, help="portion weight in grams (default: typical serving)")
    p.add_argument("--llm", choices=["none", "anthropic", "ollama"], help="override LLM_PROVIDER")
    p.add_argument("--llm-only", action="store_true", help="skip the CNN, use the LLM vision model only")
    p.add_argument("--top", type=int, default=3)
    args = p.parse_args()

    provider = args.llm or config.LLM_PROVIDER
    use_llm = llm.is_enabled(provider)
    img = Image.open(args.image)
    predictor = None if args.llm_only else Predictor()
    res = analyze(img, predictor, args.grams, provider, use_llm, args.llm_only, args.top)

    print(json.dumps(res, indent=2))
    if res.get("total_kcal") is not None:
        print(f"\n{res['food'].replace('_', ' ').title()}: ~{res['total_kcal']} kcal "
              f"for {res.get('grams')} g ({res.get('kcal_per_100g')} kcal / 100 g)")


if __name__ == "__main__":
    main()
