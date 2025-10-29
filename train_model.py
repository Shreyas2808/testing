# train_model.py
"""
Train a classifier to predict crop Type from soil/fertilizer features.

Usage:
    python train_model.py --data path/to/karnataka_city_crop_fertilizer_dataset_expanded_with_type.csv

Outputs:
    artifacts/xgb_model.pkl       (model object)
    artifacts/preprocessor.pkl    (ColumnTransformer)
    artifacts/pipeline.pkl        (full sklearn pipeline)
    artifacts/meta.json           (schema + classes + chosen model)
    reports/accuracy_summary.csv  (CV results)
"""

import argparse
import json
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

# Try to import XGBoost, fallback to HistGradientBoosting if missing
# Try to import XGBoost, fallback to HGB if missing
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

SEED = 42
np.random.seed(SEED)

# Candidate feature lists (choose intersection with actual columns)
NUM_CANDIDATES = [
    "soil_ph", "organic_carbon_pct", "sand_pct", "silt_pct", "clay_pct",
    "N_req_kg_ha", "P2O5_req_kg_ha", "K2O_req_kg_ha",
    # optional engineered or extra columns will be auto-detected if present
    "NPK_total", "N_to_P", "N_to_K", "P_to_K"
]

CAT_CANDIDATES = [
    "soil_type", "season", "irrigation", "fertilizer_type", "city"
]

TARGET = "Type"

def detect_features(df):
    num_features = [c for c in NUM_CANDIDATES if c in df.columns]
    cat_features = [c for c in CAT_CANDIDATES if c in df.columns]
    return num_features, cat_features

def build_preprocessor(num_features, cat_features):
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]) if num_features else "drop"

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]) if cat_features else "drop"

    transformers = []
    if num_features:
        transformers.append(("num", numeric_pipe, num_features))
    if cat_features:
        transformers.append(("cat", cat_pipe, cat_features))

    pre = ColumnTransformer(transformers, remainder="drop")
    return pre

def build_models(random_state=SEED):
    models = {}
    # XGBoost (if available)
    if XGB_AVAILABLE:
        models["xgboost"] = XGBClassifier(
            n_estimators=600, max_depth=6, learning_rate=0.06,
            subsample=0.9, colsample_bytree=0.85,
            random_state=random_state, use_label_encoder=False, eval_metric="mlogloss", n_jobs=-1
        )
    # HistGradientBoosting fallback
    models["hgb"] = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.08, max_depth=None, random_state=random_state
    )
    # Add a strong random forest as well
    from sklearn.ensemble import RandomForestClassifier
    models["rf"] = RandomForestClassifier(n_estimators=400, max_features="sqrt", random_state=random_state, n_jobs=-1, class_weight="balanced_subsample")
    return models

def cv_scores_for_model(pipe, X, y, skf):
    # Returns accuracy mean/std and macro-F1 mean/std
    acc = cross_val_score(pipe, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
    f1m = cross_val_score(pipe, X, y, cv=skf, scoring="f1_macro", n_jobs=-1)
    return {
        "acc_mean": float(np.mean(acc)), "acc_std": float(np.std(acc)),
        "f1_mean": float(np.mean(f1m)), "f1_std": float(np.std(f1m))
    }

def main(args):
    path = Path(args.data)
    assert path.exists(), f"Dataset not found: {path}"
    df = pd.read_csv(path)

    # Basic checks
    assert TARGET in df.columns, f"Target column '{TARGET}' not found in dataset."

    # Coerce numeric columns if present (best-effort)
    for col in df.columns:
        if col in NUM_CANDIDATES:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Detect features to use
    num_features, cat_features = detect_features(df)
    print("Detected numeric features:", num_features)
    print("Detected categorical features:", cat_features)

    # Drop rows missing the target
    df = df[ df[TARGET].notna() ].copy()
    y_raw = df[TARGET].astype(str).str.strip()
    le = LabelEncoder()
    y = le.fit_transform(y_raw)   # numeric labels for CV; store classes later

    X = df[num_features + cat_features] if (num_features + cat_features) else pd.DataFrame(index=df.index)

    # Small sanity: if no features, bail
    if X.shape[1] == 0:
        raise RuntimeError("No features detected for model training. Check dataset columns.")

    # Build preprocessor & model candidates
    pre = build_preprocessor(num_features, cat_features)
    models = build_models()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_report = []

    # Evaluate each model with pipeline (pre + model)
    for name, model in models.items():
        pipe = make_pipeline(pre, model)
        try:
            scores = cv_scores_for_model(pipe, X, y, skf)
            print(f"Model {name}: acc={scores['acc_mean']:.4f} f1_macro={scores['f1_mean']:.4f}")
            cv_report.append({
                "model": name,
                "acc_mean": scores["acc_mean"],
                "acc_std": scores["acc_std"],
                "f1_mean": scores["f1_mean"],
                "f1_std": scores["f1_std"]
            })
        except Exception as e:
            print(f"Model {name} failed during CV: {e}")
            cv_report.append({
                "model": name,
                "acc_mean": None, "acc_std": None, "f1_mean": None, "f1_std": None
            })

    report_df = pd.DataFrame(cv_report).sort_values(by=["acc_mean"], ascending=False)
    print("\n=== CV summary ===")
    print(report_df)

    # Choose best model (highest acc_mean, ignoring None)
    valid = report_df[report_df["acc_mean"].notna()]
    if valid.empty:
        raise RuntimeError("All models failed or returned no CV results.")
    best_row = valid.iloc[0]
    best_name = best_row["model"]
    print(f"\nSelected model: {best_name}")

    # Recreate chosen model instance and fit on full data
    chosen_model = models[best_name]
    full_pipe = make_pipeline(pre, chosen_model)
    print("Fitting final pipeline on entire dataset...")
    full_pipe.fit(X, y)

    # Save artifacts
    artifacts_dir = Path("artifacts"); artifacts_dir.mkdir(exist_ok=True)
    reports_dir = Path("reports"); reports_dir.mkdir(exist_ok=True)

    # Save full pipeline (convenient)
    joblib.dump(full_pipe, artifacts_dir / "pipeline.pkl")
    # Save preprocessor and model separately for compatibility with app.py
    # Extract steps
    # Note: pipeline.named_steps keys may be 'columntransformer' and 'xgbclassifier' etc.
    # We'll attempt to find the first and last steps.
    steps = list(full_pipe.named_steps.items())
    if len(steps) >= 1:
        preproc_obj = steps[0][1]
    else:
        preproc_obj = pre
    model_obj = steps[-1][1]

    joblib.dump(preproc_obj, artifacts_dir / "preprocessor.pkl")
    joblib.dump(model_obj, artifacts_dir / "xgb_model.pkl")  # app expects xgb_model.pkl name

    # Meta
    meta = {
        "target": TARGET,
        "classes": le.classes_.tolist(),
        "num_features": num_features,
        "cat_features": cat_features,
        "chosen_model": best_name,
        "xgb_available": bool(XGB_AVAILABLE)
    }
    (artifacts_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    # Save CV results
    report_df.to_csv(reports_dir / "accuracy_summary.csv", index=False)

    # Optional quick test on holdout split to show expected accuracy
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
        # fit a fresh copy to avoid reuse issues
        quick_pipe = make_pipeline(pre, chosen_model.__class__(**chosen_model.get_params()))
        quick_pipe.fit(X_train, y_train)
        y_pred = quick_pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1m = f1_score(y_test, y_pred, average="macro")
        print(f"\nHoldout test accuracy: {acc:.4f}, macro-F1: {f1m:.4f}")
        print("\nClassification report (holdout):")
        print(classification_report(y_test, y_pred, target_names=le.classes_))
    except Exception as e:
        print("Holdout test skipped due to:", e)

    print("\nSaved artifacts to 'artifacts/' and CV report to 'reports/'")
    print("Meta summary:", json.dumps(meta, indent=2))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to enriched dataset CSV")
    args = p.parse_args()
    main(args)
