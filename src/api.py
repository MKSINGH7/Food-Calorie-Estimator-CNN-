"""REST API + tiny web UI.

    uvicorn src.api:app --host 0.0.0.0 --port 8000
    open http://localhost:8000          (upload page)
    open http://localhost:8000/docs     (Swagger UI)

POST /predict   multipart: file=<image>, grams=<optional>, use_llm=<true|false>, llm_only=<true|false>
"""
import io
import logging

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

from . import calories, config, llm
from .predict import Predictor, analyze

log = logging.getLogger("food-api")
app = FastAPI(title="Food Calories CNN", version="1.0")
_predictor: Predictor | None = None
_load_error: str | None = None


@app.on_event("startup")
def _load():
    global _predictor, _load_error
    try:
        _predictor = Predictor()
        log.info("Model loaded (%d classes)", len(_predictor.class_names))
    except FileNotFoundError as e:
        _load_error = str(e)
        log.warning(_load_error)


@app.get("/")
def index():
    return FileResponse(config.ROOT / "static" / "index.html")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _predictor is not None,
        "model_error": _load_error,
        "llm_provider": config.LLM_PROVIDER,
    }


@app.get("/classes")
def classes():
    names = _predictor.class_names if _predictor else calories.food_classes()
    return {"classes": names}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    grams: float | None = Form(None),
    use_llm: bool = Form(False),
    llm_only: bool = Form(False),
):
    try:
        image = Image.open(io.BytesIO(await file.read()))
        image.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "Uploaded file is not a valid image")

    if _predictor is None and not (llm_only or (use_llm and llm.is_enabled())):
        raise HTTPException(503, _load_error or "Model not loaded. Train it first: python -m src.train")
    try:
        return analyze(
            image,
            None if llm_only else _predictor,
            grams=grams,
            use_llm=use_llm,
            llm_only=llm_only or _predictor is None,
        )
    except llm.LLMError as e:
        raise HTTPException(502, str(e))
