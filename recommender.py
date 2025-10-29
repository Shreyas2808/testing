# Recommender implementation using the provided dataset.
# - Computes crop-type similarities from numeric + categorical signals
# - Provides fertilizer->crop suggestions by frequency
# - Supports "similar crop constrained by same fertilizer"
#
# You can still drop in model artifacts (XGBoost + preprocessors), and this module
# will try to load them if present. The dataset-first path is used for similarity.

from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import os, json
import pandas as pd
import numpy as np

# ---- pick the crop-name column from the CSV ----
CROP_COL_CANDIDATES = ["crop", "Crop", "crop_name", "Crop_Name", "CROP", "CropName", "Crop Type", "CropType"]
GENERIC_TYPES = {"vegetable","fruit","fruits","cereal","cereals","pulse","pulses","oilseed","oilseeds","cash crop","cashcrop","nan"}

def _filter_output(lst, k):
    """
    Clean the list of crop suggestions:
    - Remove generic labels (cereal, pulse, etc.)
    - Remove duplicates and empty strings
    - Limit to top k
    """
    clean = [x for x in lst if x and x not in GENERIC_TYPES]
    seen = set()
    out = []
    for x in clean:
        if x not in seen:
            seen.add(x)
            out.append(x)
        if len(out) >= k:
            break
    return out

# Optional: load ML artifacts if available
try:
    import joblib  # type: ignore
except Exception:
    joblib = None

# Optional: for cosine similarity
try:
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.metrics.pairwise import cosine_similarity
except Exception as e:
    OneHotEncoder = None
    cosine_similarity = None

# ---------------- Config ----------------
# Default dataset file name (place it in project root). If not found, tries env var and /mnt/data fallback.
DEFAULT_DATASET = os.environ.get("DATA_PATH", "karnataka_city_crop_fertilizer_dataset_expanded_with_type.csv")
FALLBACK_DATASET = "/mnt/data/karnataka_city_crop_fertilizer_dataset_expanded_with_type.csv"

ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "xgb_model.pkl")
PP_PATH    = os.path.join(ARTIFACTS_DIR, "preprocessor.pkl")
LE_PATH    = os.path.join(ARTIFACTS_DIR, "label_encoder.pkl")
META_PATH  = os.path.join(ARTIFACTS_DIR, "meta.json")

# Feature columns used in the user's training script (if needed)
NUM = ['N_req_kg_ha','P2O5_req_kg_ha','K2O_req_kg_ha','soil_ph',
       'organic_carbon_pct','sand_pct','silt_pct','clay_pct']
CAT = ['soil_type','season','irrigation','fertilizer_type']
TARGET = 'Type'

# ---------------- Load dataset ----------------
def _resolve_dataset_path() -> str:
    for path in (DEFAULT_DATASET, FALLBACK_DATASET):
        if path and os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Dataset not found. Place '{DEFAULT_DATASET}' in project root or set DATA_PATH env var."
    )

_df = None
_crop_col = None

def get_df() -> pd.DataFrame:
    global _df, _crop_col
    if _df is None:
        ds_path = _resolve_dataset_path()
        df = pd.read_csv(ds_path)

        # Find a crop name column
        found = None
        for c in CROP_COL_CANDIDATES:
            if c in df.columns:
                found = c
                break

        if not found:
            # fallback: if no explicit crop column exists, attempt to synthesize from Type + maybe another col
            # but still prefer a real column if present
            found = "Type"  # last resort

        _crop_col = found

        # normalize text columns we use
        if "Type" in df.columns:
            df["Type"] = df["Type"].astype(str).str.strip().str.lower()
        df[_crop_col] = df[_crop_col].astype(str).str.strip().str.lower()

        # drop blank/na crop names
        bad_vals = {"", "nan", "none", "null"}
        df = df[~df[_crop_col].isin(bad_vals)].copy()

        _df = df
    return _df

def crop_col() -> str:
    # returns the column to treat as the unique crop name
    return _crop_col or "Type"


# ---------------- Load artifacts if present (not strictly required for current flows) ----------------
_model = _pp = _le = _meta = None
def _try_load_artifacts():
    global _model, _pp, _le, _meta
    if not joblib:
        return
    try:
        if os.path.exists(MODEL_PATH):
            _model = joblib.load(MODEL_PATH)
        if os.path.exists(PP_PATH):
            _pp = joblib.load(PP_PATH)
        if os.path.exists(LE_PATH):
            _le = joblib.load(LE_PATH)
        if os.path.exists(META_PATH):
            with open(META_PATH, "r") as f:
                _meta = json.load(f)
    except Exception:
        # Artifacts are optional for the current recommendation logic
        pass

_try_load_artifacts()

# ---------------- Build indices for recommendations ----------------
# 1) Fertilizer -> top crops (Type) by frequency
# 2) Crop (Type) -> similar crops using a representation built from dataset
_fert_to_crops: Dict[str, List[str]] = {}
_crop_to_ferts: Dict[str, List[str]] = {}
_crop_sim: Optional[pd.DataFrame] = None
_all_crops: List[str] = []

def _build_indices():
    global _fert_to_crops, _crop_to_ferts, _crop_sim, _all_crops

    df = get_df().copy()
    cc = crop_col()  # crop-name column
    if 'fertilizer_type' in df.columns:
        df['fertilizer_type'] = df['fertilizer_type'].astype(str).str.strip().str.lower()
    else:
        df['fertilizer_type'] = 'unknown'

    # Fertilizer -> crop (by frequency)
    fert_counts = df.groupby(['fertilizer_type', cc]).size().reset_index(name='count')
    fert_map = {}
    for fert, grp in fert_counts.groupby('fertilizer_type'):
        ranked = grp.sort_values('count', ascending=False)[cc].tolist()
        # remove generic categories if any slipped in
        ranked = [x for x in ranked if x not in GENERIC_TYPES]
        fert_map[fert] = ranked
    _fert_to_crops = fert_map

    # Crop -> fertilizers
    crop_ferts = df.groupby([cc, 'fertilizer_type']).size().reset_index(name='count')
    c2f = {}
    for crop, grp in crop_ferts.groupby(cc):
        ranked = grp.sort_values('count', ascending=False)['fertilizer_type'].tolist()
        c2f[crop] = ranked
    _crop_to_ferts = c2f

    # Similarity: aggregate by CROP KEY; keep `Type` as a feature (categorical) if present
    types_for_feature = []
    if 'Type' in df.columns:
        types_for_feature = ['Type']

    _all_crops = sorted(df[cc].unique().tolist())

    # numeric means per crop
    num_cols = [c for c in NUM if c in df.columns]
    num_agg = df.groupby(cc)[num_cols].mean() if num_cols else pd.DataFrame(index=_all_crops)

    # categorical frequency per crop (including 'Type' as a feature, not as key)
    cat_cols = [c for c in CAT if c in df.columns]
    cat_cols = list(dict.fromkeys(cat_cols + types_for_feature))  # include Type as feature if present

    cat_freq_parts = []
    for col in cat_cols:
        freq = (df.groupby([cc, col]).size() / df.groupby(cc).size()).reset_index(name=f'freq_{col}')
        pivot = freq.pivot(index=cc, columns=col, values=f'freq_{col}').fillna(0.0)
        pivot.columns = [f"{col}::{str(v)}" for v in pivot.columns]
        cat_freq_parts.append(pivot)

    # combine
    parts = [num_agg] + cat_freq_parts
    base = None
    for p in parts:
        if p is None or p.empty:
            continue
        base = p if base is None else base.join(p, how='outer')
    if base is None:
        base = pd.DataFrame(np.eye(len(_all_crops)), index=_all_crops, columns=[f"feat_{i}" for i in range(len(_all_crops))])

    base = base.fillna(0.0)

    # standardize & similarity
    std = base.std(axis=0).replace(0, 1.0)
    base_norm = (base - base.mean(axis=0)) / std

    try:
        from sklearn.metrics.pairwise import cosine_similarity
        sim = cosine_similarity(base_norm.values)
    except Exception:
        X = base_norm.values
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms==0] = 1.0
        Xn = X / norms
        sim = Xn @ Xn.T

    _crop_sim = pd.DataFrame(sim, index=base_norm.index, columns=base_norm.index)


# ---------- Enhancements for robust suggestions ----------
from difflib import get_close_matches
_global_top_crops = []

def _build_global_top():
    global _global_top_crops
    df = get_df().copy()
    cc = crop_col()
    counts = df[cc].value_counts()
    ranked = counts.index.tolist()
    _global_top_crops = [x for x in ranked if x not in GENERIC_TYPES]


def resolve_crop_name(crop: str) -> str | None:
    if not crop:
        return None
    c = normalize(crop)
    if c in _all_crops:
        return c
    match = get_close_matches(c, _all_crops, n=1, cutoff=0.6)
    return match[0] if match else None

_build_indices()
_build_global_top()

# ---------------- Public API ----------------
def normalize(text: str) -> str:
    return (text or "").strip().lower()

def crops_supported_by_fertilizer(fertilizer: str) -> list:
    key = normalize(fertilizer)
    return _fert_to_crops.get(key, [])

def _neighbors(crop: str, k: int = 10) -> List[str]:
    crop_n = normalize(crop)
    if _crop_sim is None or crop_n not in _crop_sim.index:
        return []
    # rank by similarity, exclude self
    scores = _crop_sim.loc[crop_n].sort_values(ascending=False)
    nbrs = [c for c in scores.index if c != crop_n]
    return nbrs[:k]

def suggest_by_crop(crop: str, k: int = 10) -> Dict[str, Any]:
    rc = resolve_crop_name(crop)
    if rc is None:
        # Crop name not found — fallback to top crops, filter out generics
        return {
            "basis": "crop",
            "input": crop,
            "suggestions": _filter_output(_global_top_crops, k)
        }

    nbrs = _neighbors(rc, k=50)  # get more neighbors, then filter down
    if not nbrs:
        # no neighbors — fallback to top crops except itself
        fallback = [c for c in _global_top_crops if c != rc]
        return {
            "basis": "crop",
            "input": crop,
            "suggestions": _filter_output(fallback, k)
        }

    # ✅ normal case — filter neighbor list to remove generic labels
    return {
        "basis": "crop",
        "input": crop,
        "suggestions": _filter_output(nbrs, k)
    }

def suggest_by_crop_with_fertilizer(crop: str, fertilizer: str, k: int = 10) -> Dict[str, Any]:
    rc = resolve_crop_name(crop)
    if rc is None:
        # unknown crop: if fertilizer provided, use that; else global
        if fertilizer:
            allowed = crops_supported_by_fertilizer(fertilizer)
            return {"basis": "crop+fertilizer", "input": {"crop": crop, "fertilizer": fertilizer}, "suggestions": allowed[:k]}
        return {"basis": "crop+fertilizer", "input": {"crop": crop, "fertilizer": fertilizer}, "suggestions": _global_top_crops[:k]}

    if not fertilizer:
        return suggest_by_crop(rc, k=k)

    allowed = set(crops_supported_by_fertilizer(fertilizer))
    # If allowed is empty, fall back to neighbors without filter
    if not allowed:
        base = _neighbors(rc, k=50)
        if not base:
            fallback = [c for c in _global_top_crops if c != rc][:k]
            return {"basis": "crop+fertilizer", "input": {"crop": crop, "fertilizer": fertilizer}, "suggestions": fallback}
        return {"basis": "crop+fertilizer", "input": {"crop": crop, "fertilizer": fertilizer}, "suggestions": base[:k]}

    nbrs = _neighbors(rc, k=50)
    filtered = [c for c in nbrs if c in allowed]
    if not filtered:
        # Fallback to top crops for this fertilizer
        fert_top = list(allowed)  # preserve order if set? allowed is set; get original list again
        fert_top = crops_supported_by_fertilizer(fertilizer)  # ordered list
        return {"basis": "crop+fertilizer", "input": {"crop": crop, "fertilizer": fertilizer}, "suggestions": fert_top[:k]}

    return {"basis": "crop+fertilizer", "input": {"crop": crop, "fertilizer": fertilizer}, "suggestions": filtered[:k]}

# add this import near the top of recommender.py (if not already present)
from difflib import get_close_matches

# Replace the old suggest_by_fertilizer with this improved version:
def suggest_by_fertilizer(fertilizer: str, k: int = 10) -> Dict[str, Any]:
    """
    Robust fertilizer -> crop suggestion:
    - Uses exact normalized key when available
    - Tries substring matches (e.g. "urea" -> "urea + ssp + mop")
    - Falls back to fuzzy close matches among fertilizer keys
    - Final fallback: return global top crops
    """
    # defensive: if no input, return global top crops
    if not fertilizer:
        return {"basis": "fertilizer", "input": fertilizer, "suggestions": _global_top_crops[:k]}

    key = normalize(fertilizer)

    # 1) exact match (fast)
    try:
        if key in _fert_to_crops and _fert_to_crops[key]:
            return {"basis": "fertilizer-exact", "input": fertilizer, "suggestions": _fert_to_crops[key][:k]}
    except Exception:
        # if mapping not built yet, fall back to crops_supported_by_fertilizer
        pass

    # 2) substring match against fertilizer keys (most useful for combined strings)
    substring_matches = []
    for fert_key in _fert_to_crops.keys():
        if not fert_key:
            continue
        # if user typed a token contained in a key (e.g., "urea" in "urea + ssp + mop")
        if key and key in fert_key:
            substring_matches.append((_fert_to_crops[fert_key], fert_key))
        # or if fertilizer key is contained in user input (user typed "urea+mop")
        elif fert_key in key:
            substring_matches.append((_fert_to_crops[fert_key], fert_key))

    if substring_matches:
        seen = set()
        out = []
        for crop_list, fert_key in substring_matches:
            for c in crop_list:
                if c not in seen:
                    seen.add(c)
                    out.append(c)
                if len(out) >= k:
                    break
            if len(out) >= k:
                break
        return {"basis": "fertilizer-substr", "input": fertilizer, "suggestions": out[:k]}

    # 3) fuzzy match the user's key against known fertilizer keys
    keys = list(_fert_to_crops.keys())
    close = get_close_matches(key, keys, n=3, cutoff=0.5)
    if close:
        seen = set()
        out = []
        for ck in close:
            for c in _fert_to_crops.get(ck, []):
                if c not in seen:
                    seen.add(c)
                    out.append(c)
                if len(out) >= k:
                    break
            if len(out) >= k:
                break
        return {"basis": "fertilizer-fuzzy", "input": fertilizer, "suggestions": out[:k]}

    # 4) last-resort: try the existing helper crops_supported_by_fertilizer (maybe it does something)
    try:
        lst = crops_supported_by_fertilizer(fertilizer)
        if lst:
            return {"basis": "fertilizer-helper-fallback", "input": fertilizer, "suggestions": lst[:k]}
    except Exception:
        pass

    # 5) absolute fallback: global top crops
    return {"basis": "fertilizer-fallback", "input": fertilizer, "suggestions": _global_top_crops[:k]}


# Convenience: expose known crops/fertilizers for dropdowns/autocomplete if needed
def known_crops() -> List[str]:
    return _all_crops

def known_fertilizers() -> List[str]:
    return sorted(list(_fert_to_crops.keys()))