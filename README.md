# Food Calories CNN

Take a photo of a dish → a CNN recognises the food → the app looks up its calories
(per 100 g and for your portion) → optionally an LLM (local **or** online) adds nutrition
advice, and steps in when the CNN is unsure.

```
photo ──► CNN (MobileNetV2) ──► food + confidence ──► calories table ──► kcal for your portion
                │                                                              │
                └── confidence < threshold? ──► LLM vision (Claude / Ollama)   └─► LLM nutrition advice (optional)
```

## Project layout

```
food_calories_cnn/
├── data/calories.json        kcal/100 g + typical serving per food  (also defines the classes)
├── src/
│   ├── config.py             paths + .env settings
│   ├── prepare_data.py       download Food-101 subset / split your own photos
│   ├── data.py               tf.data loaders
│   ├── model.py              MobileNetV2 (default) or small CNN
│   ├── train.py              2-phase training, plots, report
│   ├── predict.py            CLI + analyze() pipeline
│   ├── llm.py                Anthropic (online) / Ollama (local) client
│   └── api.py                FastAPI server
├── static/index.html         upload web page served at /
├── scripts/make_dummy_dataset.py   tiny fake dataset for a 1-minute smoke test
├── .env.example  requirements.txt  .gitignore
└── models/ outputs/ dataset/  (created/filled by the steps below)
```

## 1. Install

Requirements: **Python 3.10 – 3.13**, ~6 GB free disk for the dataset, internet for the first run
(pip packages, ImageNet weights, Food-101). A GPU is optional; CPU works, just slower.

```bash
cd food_calories_cnn
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env                 # Windows: copy .env.example .env
```

## 2. Quick smoke test (1–2 minutes, no downloads)

Checks that your install works end to end using a synthetic dataset (accuracy is meaningless):

```bash
python scripts/make_dummy_dataset.py --classes 4
python -m src.train --data-dir dataset_dummy --arch cnn --epochs 6 --img-size 96
python -m src.predict dataset_dummy/validation/apple_pie/000.png
```

## 3. Get the real dataset

**Food-101 (default; 25 classes listed in `data/calories.json`)** – downloads ~4.6 GB once and keeps only the
classes you need (default 600 train + 150 validation images per class):

```bash
python -m src.prepare_data
# smaller / faster:  python -m src.prepare_data --train-per-class 300 --val-per-class 80
# specific foods:    python -m src.prepare_data --classes pizza sushi hamburger ice_cream
```
Already have Food-101 extracted? `python -m src.prepare_data --food101-dir /path/to/food-101`
(If the download is blocked on your network, download `food-101.tar.gz` from
<https://data.vision.ee.ethz.ch/cvl/food-101/> in a browser and place it in the project folder.)

**Your own photos** – put them in `my_photos/<class_name>/*.png`, then
`python -m src.prepare_data --source-dir my_photos`, and add each new class to `data/calories.json`.

Result: `dataset/train/<class>/…` and `dataset/validation/<class>/…`

## 4. Train

```bash
python -m src.train
```
Phase 1 trains the new classifier head, phase 2 fine-tunes the top of MobileNetV2 with a lower learning rate. Early
stopping + best-weights restore are built in. Useful flags: `--epochs`, `--finetune-epochs`, `--batch-size`,
`--arch cnn`, `--img-size 160` (faster on CPU).

Outputs: `models/food_classifier.keras`, `models/model_meta.json`, and in `outputs/`: `accuracy.png`, `loss.png`,
`confusion_matrix.png`, `classification_report.txt`, `metrics.json`.
**Compare `outputs/accuracy.png` with the old plot** – the train/validation gap should be far smaller.
Expect a CPU run of roughly 30–90 minutes for the default settings; a GPU (or Google Colab) takes minutes.
<img width="557" height="700" alt="Screenshot 2026-10-10 065108" src="https://github.com/user-attachments/assets/b2685cbc-5b8c-4327-82fd-6bc4dae4caa1" />

## 5. Predict

```bash
python -m src.predict photo.png                 # typical serving
python -m src.predict photo.png --grams 250     # your own portion size
```

## 6. Run the API + web page

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```
* Web page: <http://localhost:8000> – upload a photo, optional grams, optional "Use LLM"
* Swagger docs: <http://localhost:8000/docs>
* From code:
  ```bash
  curl -F file=@photo.png -F grams=250 -F use_llm=false http://localhost:8000/predict
  ```
  Endpoints: `GET /health`, `GET /classes`, `POST /predict` (`file`, `grams`, `use_llm`, `llm_only`).

## 7. Add an LLM (optional) – online or local

The CNN works fully offline. An LLM adds two things: a short **nutrition advice** text, and a **vision fallback**
when the CNN's top confidence is below `CONF_THRESHOLD` (default 0.50) – e.g. for a dish that is not among your classes.
With `llm_only` you can even skip training and let the LLM identify the food alone.

### Option A – Online: Anthropic Claude API
1. Create an API key at <https://console.anthropic.com> (API usage is billed separately from claude.ai plans).
2. Edit `.env`:
   ```ini
   LLM_PROVIDER=anthropic
   ANTHROPIC_API_KEY=sk-ant-...
   ANTHROPIC_MODEL=claude-sonnet-5-5
   ```
3. Use it:
   ```bash
   python -m src.predict photo.png --llm anthropic
   python -m src.predict photo.png --llm anthropic --llm-only     # no trained CNN needed
   ```
   In the web page tick **Use LLM**; in the API send `use_llm=true`.

### Option B – Local: Ollama (private, free, offline after download)
1. Install Ollama from <https://ollama.com>, then pull a vision-capable model and keep the server running:
   ```bash
   ollama pull llava          # or: llama3.2-vision, qwen2.5vl, gemma3
   ollama serve               # skip if the desktop app already runs it
   ```
2. Edit `.env`:
   ```ini
   LLM_PROVIDER=ollama
   OLLAMA_URL=http://localhost:11434
   OLLAMA_MODEL=llama3.2:1b
   ```
3. `python -m src.predict photo.png --llm ollama`

Notes: local vision models are less accurate at calorie estimation than the CNN-plus-table route or large online
models – treat their numbers as rough. Never commit `.env` (it is in `.gitignore`).

## Input
<img width="1912" height="862" alt="Screenshot 2026-10-10 080254" src="https://github.com/user-attachments/assets/07455bad-9c41-4e29-b9a1-5e7410f4b262" />

## Output
<img width="1896" height="857" alt="Screenshot 2026-10-10 080343" src="https://github.com/user-attachments/assets/96421d08-643f-4565-892c-caa429a444d3" />

## Workflow of the whole project

```
 ┌─────────────── ONE-TIME (offline) ───────────────┐      ┌──────────── EVERY REQUEST ────────────┐
 │                                                   │      │                                       │
 │ 1 calories.json  ─ defines the food classes       │      │ A  user uploads photo (web/API/CLI)   │
 │ 2 prepare_data   ─ Food-101 / own photos          │      │ B  resize 224×224, feed raw pixels    │
 │        └► dataset/train + dataset/validation      │      │ C  CNN → top-3 foods + confidence     │
 │ 3 train          ─ phase 1: head, phase 2: tune   │      │ D  confidence ≥ threshold?            │
 │        └► models/food_classifier.keras            │ ───► │     yes → calories.json lookup        │
 │           models/model_meta.json                  │      │           kcal = kcal/100g × grams    │
 │           outputs/ (curves, report, matrix)       │      │     no  → LLM vision estimate (if on) │
 │ 4 check accuracy.png: val close to train?         │      │ E  optional LLM nutrition advice      │
 └───────────────────────────────────────────────────┘      │ F  JSON / web page / console output   │
                                                            └───────────────────────────────────────┘
```

## Improving accuracy further
* More images per class (`--train-per-class 750`, the Food-101 maximum) and more epochs.
* Larger backbone: swap `MobileNetV2` in `src/model.py` for `EfficientNetB0` (note its own pre-processing).
* Check `outputs/confusion_matrix.png` to see which foods get confused and add data for them.
* Calorie accuracy is limited by portion size: always pass `--grams` / the grams field when you know the weight.
