from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from datetime import datetime
import json
import os
import pandas as pd
# Market price helpers
from market_ranking import load_market_prices, rank_crops_by_city

# ML integration
from recommender import suggest_by_fertilizer, suggest_by_crop_with_fertilizer

app = Flask(__name__)
app.config["SECRET_KEY"] = "change-me-in-prod"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///app.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# --------------------
# Load market dataset once at startup
# --------------------
MARKET_CSV_PATH = os.path.join('data', 'Market_Price_Dataset_2024_2025.csv')
market_df = load_market_prices(MARKET_CSV_PATH)

# --- add near top of app.py, after market_df is loaded ---
import joblib, json
PIPELINE_PATH = "artifacts/pipeline.pkl"
META_PATH = "artifacts/meta.json"

# load saved pipeline + meta (fail loudly with helpful message)
try:
    PIPE = joblib.load(PIPELINE_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load pipeline at {PIPELINE_PATH}: {e}")

try:
    META = json.load(open(META_PATH, "r", encoding="utf-8"))
except Exception as e:
    raise RuntimeError(f"Failed to load meta.json at {META_PATH}: {e}")

# We'll need the recommender dataset to list actual crop names for the predicted Type
from recommender import get_df as get_recommender_df, crop_col as recommender_crop_col
# rank_crops_by_city and market_df are already present earlier in your app.py

# --------------------
# Helper functions
# --------------------
def _safe_split_csv(s: str):
    if not s:
        return []
    return [x.strip() for x in s.split(',') if x.strip()]

def _ensure_suggestions_dict(suggestions_obj):
    """
    Template expects suggestions.suggestions.
    Wrap list or dict properly so template won't break.
    """
    if isinstance(suggestions_obj, dict) and 'suggestions' in suggestions_obj:
        return suggestions_obj
    if isinstance(suggestions_obj, list):
        return {'suggestions': suggestions_obj}
    try:
        if isinstance(suggestions_obj, dict):
            for _, v in suggestions_obj.items():
                if isinstance(v, list):
                    return {'suggestions': v}
    except Exception:
        pass
    return {'suggestions': []}

# --------------------
# Database Models
# --------------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120))
    email = db.Column(db.String(255), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class UserQuery(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    query_type = db.Column(db.String(50), nullable=False)  # "crop" or "fertilizer"
    input_text = db.Column(db.String(255), nullable=False)
    result_json = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# --------------------
# Auth helpers
# --------------------
def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to access this page.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapper

def current_user():
    uid = session.get("user_id")
    if not uid:
        return None
    return db.session.get(User, uid)

# --------------------
# Auth Routes
# --------------------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = request.form.get("name","").strip()
        email = request.form.get("email","").lower().strip()
        password = request.form.get("password","")

        if not email or not password:
            flash("Email and password are required.", "danger")
            return render_template("signup.html")

        if User.query.filter_by(email=email).first():
            flash("Email already registered. Try logging in.", "warning")
            return redirect(url_for("login"))

        user = User(
            name=name or None,
            email=email,
            password_hash=generate_password_hash(password)
        )
        db.session.add(user)
        db.session.commit()
        flash("Account created! Please log in.", "success")
        return redirect(url_for("login"))
    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email","").lower().strip()
        password = request.form.get("password","")
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password_hash, password):
            session["user_id"] = user.id
            flash("Welcome back!", "success")
            return redirect(url_for("dashboard"))
        flash("Invalid credentials.", "danger")
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    session.clear()
    flash("Logged out.", "info")
    return redirect(url_for("login"))

# --------------------
# App Routes
# --------------------
@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))

@app.route("/dashboard", methods=["GET"])
@login_required
def dashboard():
    user = current_user()
    recent = (UserQuery.query
              .filter_by(user_id=user.id)
              .order_by(UserQuery.created_at.desc())
              .limit(5).all())
    return render_template("dashboard.html", user=user, recent=recent)

@app.route("/recommend/crop", methods=["POST"])
@login_required
def recommend_by_crop():
    user = current_user()
    crop = request.form.get("crop","").strip()
    fertilizer_for_crop = request.form.get("fertilizer_for_crop","").strip()

    if not crop:
        flash("Please enter a crop.", "warning")
        return redirect(url_for("dashboard"))

    suggestions = suggest_by_crop_with_fertilizer(crop, fertilizer_for_crop)

    uq = UserQuery(
        user_id=user.id,
        query_type="crop",
        input_text=crop,
        result_json=json.dumps(suggestions, ensure_ascii=False)
    )
    db.session.add(uq)
    db.session.commit()

    suggestions_view = _ensure_suggestions_dict(suggestions)

    flash("Here are crop suggestions based on similarity.", "success")
    return render_template(
        "recommend.html",
        user=user,
        input_label="Input Crop",
        input_value=crop,
        suggestions=suggestions_view
    )

@app.route("/recommend/fertilizer", methods=["POST"])
@login_required
def recommend_by_fertilizer():
    user = current_user()
    fertilizer = request.form.get("fertilizer","").strip()

    if not fertilizer:
        flash("Please enter a fertilizer (e.g., Urea, MOP).", "warning")
        return redirect(url_for("dashboard"))

    suggestions = suggest_by_fertilizer(fertilizer)

    uq = UserQuery(
        user_id=user.id,
        query_type="fertilizer",
        input_text=fertilizer,
        result_json=json.dumps(suggestions, ensure_ascii=False)
    )
    db.session.add(uq)
    db.session.commit()

    suggestions_view = _ensure_suggestions_dict(suggestions)

    flash("Here are crop suggestions based on fertilizer.", "success")
    return render_template(
        "recommend.html",
        user=user,
        input_label="Fertilizer",
        input_value=fertilizer,
        suggestions=suggestions_view
    )

@app.route("/rank-market", methods=["POST"])
@login_required
def rank_market():
    user = current_user()
    city = request.form.get("city", "").strip()
    crops_csv = request.form.get("top_suggestions", "")
    crops = _safe_split_csv(crops_csv)

    if not city or not crops:
        flash("Missing city or top suggestions.", "warning")
        return redirect(url_for("dashboard"))

    ranked_df = rank_crops_by_city(crops, city, df=market_df, price_col='modal price')
    ranked = ranked_df.to_dict(orient='records')

    return render_template(
        "recommend.html",
        user=user,
        input_label="City",
        input_value=city,
        suggestions={'suggestions': crops},
        ranked_by_price=ranked,
        city=city
    )

# --------------------
# CLI Helper
# --------------------
@app.cli.command("init-db")
def init_db():
    """Initialize the SQLite database."""
    db.create_all()
    print("✅ Database initialized: app.db")

@app.route("/ml-recommend", methods=["GET", "POST"])
@login_required
def ml_recommend():
    user = current_user()

    # Helper: safe lower/strip
    def _norm_str(s):
        return (s or "").strip().lower()

    # numeric columns we need to fill (same as training)
    NUM_COLS = [
        "soil_ph", "organic_carbon_pct", "sand_pct", "silt_pct", "clay_pct",
        "N_req_kg_ha", "P2O5_req_kg_ha", "K2O_req_kg_ha"
    ]

    if request.method == "POST":
        city = request.form.get("city", "").strip()
        soil_type = request.form.get("soil_type", "").strip()
        season = request.form.get("season", "").strip()
        fertilizer_type = request.form.get("fertilizer_type", "").strip()

        if not city or not soil_type or not season or not fertilizer_type:
            flash("Please fill city, soil type, season and fertilizer type.", "warning")
            return redirect(url_for("ml_recommend"))

        # Load recommender dataset to compute representative numeric values
        try:
            rec_df = get_recommender_df().copy()
        except Exception as e:
            flash(f"Failed to load recommender dataset: {e}", "danger")
            return redirect(url_for("dashboard"))

        # normalize columns for matching
        for c in ["city", "soil_type", "season", "fertilizer_type", "irrigation"]:
            if c in rec_df.columns:
                rec_df[c + "__norm"] = rec_df[c].astype(str).str.lower().str.strip()
            else:
                rec_df[c + "__norm"] = ""

        city_n = _norm_str(city)
        soil_n = _norm_str(soil_type)
        season_n = _norm_str(season)
        fert_n = _norm_str(fertilizer_type)

        # Strategy: try increasingly relaxed filters to find rows similar to user input
        filters = [
            (rec_df["city__norm"] == city_n) & (rec_df["soil_type__norm"].str.contains(soil_n, na=False)) & (rec_df["season__norm"] == season_n) & (rec_df["fertilizer_type__norm"].str.contains(fert_n, na=False)),
            (rec_df["city__norm"] == city_n) & (rec_df["soil_type__norm"].str.contains(soil_n, na=False)) & (rec_df["season__norm"] == season_n),
            (rec_df["city__norm"] == city_n) & (rec_df["season__norm"] == season_n),
            (rec_df["soil_type__norm"].str.contains(soil_n, na=False)) & (rec_df["season__norm"] == season_n),
            (rec_df["city__norm"] == city_n),
            # final fallback: full dataset
            (rec_df.index == rec_df.index)
        ]

        selected = pd.DataFrame()
        for f in filters:
            cand = rec_df[f].copy()
            if not cand.empty:
                selected = cand
                break

        # If nothing matched (shouldn't happen due to last fallback), selected will be full rec_df
        if selected.empty:
            selected = rec_df.copy()

        # Compute medians for numeric features and mode for irrigation
        imputed = {}
        for col in NUM_COLS:
            if col in selected.columns:
                val = pd.to_numeric(selected[col], errors="coerce").median()
                if pd.isna(val):
                    # fallback to global median
                    val = pd.to_numeric(rec_df[col], errors="coerce").median()
                imputed[col] = float(val) if not pd.isna(val) else None
            else:
                imputed[col] = None

        # irrigation: use mode from selected rows, fallback to global mode or 'Irrigated'
        irrigation_val = None
        if "irrigation" in selected.columns:
            irrigation_val = selected["irrigation"].mode().iloc[0] if not selected["irrigation"].mode().empty else None
        if irrigation_val is None and "irrigation" in rec_df.columns:
            irrigation_val = rec_df["irrigation"].mode().iloc[0] if not rec_df["irrigation"].mode().empty else None
        irrigation_val = irrigation_val if irrigation_val else "Irrigated"

        # Build model input DataFrame (must match training columns)
        model_input = {
            "soil_ph": imputed.get("soil_ph"),
            "organic_carbon_pct": imputed.get("organic_carbon_pct"),
            "sand_pct": imputed.get("sand_pct"),
            "silt_pct": imputed.get("silt_pct"),
            "clay_pct": imputed.get("clay_pct"),
            "N_req_kg_ha": imputed.get("N_req_kg_ha"),
            "P2O5_req_kg_ha": imputed.get("P2O5_req_kg_ha"),
            "K2O_req_kg_ha": imputed.get("K2O_req_kg_ha"),
            "soil_type": soil_type,
            "season": season,
            "irrigation": irrigation_val,
            "fertilizer_type": fertilizer_type,
            "city": city
        }

        input_df = pd.DataFrame([model_input])

        # Predict using pipeline
        try:
            pred_idx = PIPE.predict(input_df)[0]
            predicted_type = META["classes"][int(pred_idx)]
        except Exception as e:
            flash(f"Prediction failed: {e}", "danger")
            return redirect(url_for("ml_recommend"))

        # model confidence (if available)
        model_confidence = None
        try:
            proba = PIPE.predict_proba(input_df)[0]
            model_confidence = float(max(proba))
        except Exception:
            model_confidence = None

        # Gather candidate crop names of that Type
        try:
            rec_df_full = get_recommender_df()
            crop_col_name = recommender_crop_col()
            candidates = rec_df_full[rec_df_full["Type"].astype(str).str.lower() == str(predicted_type).lower()][crop_col_name].dropna().unique().tolist()
            candidates = [c.strip() for c in candidates if c and str(c).strip()]
        except Exception as e:
            flash(f"Failed to fetch candidate crops for Type '{predicted_type}': {e}", "danger")
            return redirect(url_for("ml_recommend"))

        if not candidates:
            flash(f"No candidate crops found for predicted type: {predicted_type}", "warning")
            return render_template("recommend.html", user=user, input_label="Predicted Type", input_value=predicted_type, suggestions={'suggestions': []}, city=city)

        # Rank candidates by market modal price for the selected city
        try:
            ranked_df = rank_crops_by_city(candidates, city, df=market_df, price_col='modal price')
            ranked = ranked_df.to_dict(orient='records')
        except Exception as e:
            flash(f"Ranking failed: {e}", "danger")
            return render_template("recommend.html", user=user, input_label="Predicted Type", input_value=predicted_type, suggestions={'suggestions': candidates}, city=city)

        # Pass the imputed values + confidence to the template so user can see what was used
        used_imputed = model_input.copy()
        # ensure numeric formatting
        for k in NUM_COLS:
            if used_imputed.get(k) is not None:
                used_imputed[k] = round(float(used_imputed[k]), 3)

        return render_template(
            "recommend.html",
            user=user,
            input_label="Predicted Type",
            input_value=predicted_type,
            suggestions={'suggestions': candidates},
            ranked_by_price=ranked,
            city=city,
            used_imputed=used_imputed,
            model_confidence=model_confidence
        )

    # GET -> render the compact form (template below)
    return render_template("ml_predict.html", user=user)

# --------------------
# Entrypoint
# --------------------
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(host="0.0.0.0", port=5000, debug=True)