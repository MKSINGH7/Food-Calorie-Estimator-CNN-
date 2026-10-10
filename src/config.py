"""Central configuration: paths, defaults and environment variables."""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv is optional at runtime
    def load_dotenv(*_a, **_k):
        return False

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# ---- paths -----------------------------------------------------------------
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "dataset"))
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
MODEL_PATH = MODELS_DIR / "food_classifier.keras"
META_PATH = MODELS_DIR / "model_meta.json"  # class names, image size, architecture
CALORIES_PATH = ROOT / "data" / "calories.json"

# ---- training defaults -------------------------------------------------------
IMG_SIZE = 224
BATCH_SIZE = 32

# ---- LLM ---------------------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "none").strip().lower()  # none|anthropic|ollama
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llava")
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.50"))
