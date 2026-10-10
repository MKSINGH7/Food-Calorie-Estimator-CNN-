"""Optional LLM layer: online (Anthropic Claude API) or local (Ollama).

Used for two things:
  1. ``advice()``            short nutrition comment on the CNN's result (text only)
  2. ``estimate_from_image`` vision fallback when the CNN is unsure (or not trained yet)
"""
import base64
import io
import json
import re

import requests
from PIL import Image

from . import config


class LLMError(RuntimeError):
    pass


def is_enabled(provider: str | None = None) -> bool:
    return (provider or config.LLM_PROVIDER) in ("anthropic", "ollama")


def _jpeg_b64(img: Image.Image, max_side=1024) -> str:
    img = img.convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def _ask_anthropic(prompt: str, image_b64: str | None) -> str:
    if not config.ANTHROPIC_API_KEY:
        raise LLMError("ANTHROPIC_API_KEY is not set (put it in .env)")
    import anthropic

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    content = []
    if image_b64:
        content.append(
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64}}
        )
    content.append({"type": "text", "text": prompt})
    try:
        msg = client.messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=600,
            messages=[{"role": "user", "content": content}],
        )
    except Exception as e:  # network, auth, rate limit...
        raise LLMError(f"Anthropic API error: {e}") from e
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()


def _ask_ollama(prompt: str, image_b64: str | None) -> str:
    message = {"role": "user", "content": prompt}
    if image_b64:
        message["images"] = [image_b64]
    try:
        r = requests.post(
            f"{config.OLLAMA_URL}/api/chat",
            json={"model": config.OLLAMA_MODEL, "messages": [message], "stream": False},
            timeout=180,
        )
        r.raise_for_status()
        return r.json()["message"]["content"].strip()
    except Exception as e:
        raise LLMError(
            f"Ollama error ({config.OLLAMA_URL}, model '{config.OLLAMA_MODEL}'): {e}. "
            "Is `ollama serve` running and the model pulled?"
        ) from e


def ask(prompt: str, image: Image.Image | None = None, provider: str | None = None) -> str:
    provider = provider or config.LLM_PROVIDER
    image_b64 = _jpeg_b64(image) if image is not None else None
    if provider == "anthropic":
        return _ask_anthropic(prompt, image_b64)
    if provider == "ollama":
        return _ask_ollama(prompt, image_b64)
    raise LLMError("LLM_PROVIDER must be 'anthropic' or 'ollama'")


def advice(food: str, kcal_per_100g, grams, total_kcal, provider: str | None = None) -> str:
    prompt = (
        f"A food-recognition model identified a dish as '{food.replace('_', ' ')}' "
        f"(~{kcal_per_100g} kcal per 100 g; portion ~{grams} g, ~{total_kcal} kcal). "
        "In 2-3 short sentences give practical, non-judgemental nutrition advice "
        "(what to balance it with, portion tips). Do not give medical advice."
    )
    return ask(prompt, provider=provider)


_VISION_PROMPT = (
    "Look at this photo of food. Identify the main dish and estimate its calories. "
    "Reply with ONLY a JSON object, no other text, with keys: "
    '"food" (short name, lower case), "kcal_per_100g" (number), '
    '"estimated_grams" (number, the visible portion), "total_kcal" (number), '
    '"confidence" (0-1), "notes" (one short sentence).'
)


def _extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise LLMError(f"LLM did not return JSON: {text[:200]!r}")
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise LLMError(f"Could not parse LLM JSON: {e}") from e


def estimate_from_image(image: Image.Image, provider: str | None = None) -> dict:
    data = _extract_json(ask(_VISION_PROMPT, image=image, provider=provider))
    return {
        "food": str(data.get("food", "unknown")).strip().lower().replace(" ", "_"),
        "kcal_per_100g": data.get("kcal_per_100g"),
        "grams": data.get("estimated_grams"),
        "total_kcal": data.get("total_kcal"),
        "confidence": data.get("confidence"),
        "notes": data.get("notes", ""),
    }
