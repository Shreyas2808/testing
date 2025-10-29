# Flask Crop Recommender (Starter)

Features:
- User signup/login (SQLite, hashed passwords)
- Dashboard with two flows:
  - Enter planting crop ➜ get similar crops
  - Enter fertilizer (e.g., Urea, MOP) ➜ get crop suggestions
- Query history (last 5)
- Easy ML integration points in `recommender.py`

## Run locally
```bash
python -m venv .venv && . .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
python app.py  # first run will create app.db automatically
# open http://127.0.0.1:5000
```

Or with Flask CLI:
```bash
flask --app app.py init-db
flask --app app.py run --debug
```

## Where to plug in your ML
Edit `recommender.py`:
- Load your model(s)/artifacts at import time.
- Implement `suggest_by_crop(crop)` and `suggest_by_fertilizer(fertilizer)` to return:
```python
{
  "basis": "crop" | "fertilizer",
  "input": "<original input>",
  "suggestions": ["item1", "item2", ...]
}
```
Return up to ~5–10 items for a clean UI.

## Notes
- Change `SECRET_KEY` in `app.py` for production.
- This is a minimal starter; add validation, rate limiting, and CSRF protection before going live.


## ML/Data Integration (Already Wired)
This starter uses your dataset to:
- Build **fertilizer → crop** frequency tables
- Compute **crop-to-crop similarity** (numeric means + categorical frequency embeddings, cosine similarity)
- Support **"similar crop with same fertilizer"** filtering

**Dataset path**
Place `karnataka_city_crop_fertilizer_dataset_expanded_with_type.csv` in the project root, or set env var:
```
export DATA_PATH=/absolute/path/to/karnataka_city_crop_fertilizer_dataset_expanded_with_type.csv
```
It will also try a fallback at `/mnt/data/...` if available.

**Optional artifacts**
If you run your `train_and_export.py`, put generated files in `artifacts/`:
- `xgb_model.pkl`, `preprocessor.pkl`, `label_encoder.pkl`, `meta.json`
The app does not require them for the current recommendation flows, but will load them if present for future use.

