
import pandas as pd
import re

def _norm(s):
    s = str(s).lower()
    s = re.sub(r'\([^)]*\)', ' ', s)
    s = s.replace('/', ' ').replace('-', ' ')
    s = re.sub(r'[^a-z\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

_ALIAS_MAP = {
    "blackgram (urad)": ["blackgram", "black gram", "urad", "urad dal"],
    "ragi (finger millet)": ["ragi", "finger millet", "nachni", "ragulu"],
    "sorghum (jowar)": ["sorghum", "jowar", "jwari"],
    "brinjal (eggplant)": ["brinjal", "eggplant", "baingan", "aubergine"],
    "ridge gourd": ["ridge gourd", "turai", "beerakaya", "peerkangai"],
    "chilli": ["chilli", "chili", "chilies", "green chilli", "red chilli"],
    "coriander": ["coriander", "dhaniya", "coriander leaves"],
    "spinach": ["spinach", "palak"],
    "cucumber": ["cucumber", "kheera"],
    "cabbage": ["cabbage", "patta gobi"],
}

def load_market_prices(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    if set(['year','month']).issubset(df.columns):
        df['__dt'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str) + '-01', errors='coerce')
    else:
        df['__dt'] = pd.NaT
    df['crop_norm'] = df['crop'].apply(_norm)
    df['city_norm'] = df['city'].str.lower().str.strip()
    return df

def _match_rows_for_crop(df, crop_name):
    aliases = _ALIAS_MAP.get(str(crop_name).lower(), [crop_name])
    aliases_norm = [_norm(a) for a in aliases]
    mask = False
    for a in aliases_norm:
        mask = mask | df['crop_norm'].str.contains(a)
    return df[mask]

def rank_crops_by_city(crops, city, df, price_col='modal price'):
    # Handle "modal price" vs "modal_price" etc.
    if price_col not in df.columns:
        candidates = [c for c in df.columns if c.replace(' ', '') == price_col.replace(' ', '')]
        if candidates:
            price_col = candidates[0]
        else:
            raise ValueError(f"Price column '{price_col}' not found. Found columns: {df.columns.tolist()}")
    city_norm = city.strip().lower()
    out_rows = []
    for crop in crops:
        sub = _match_rows_for_crop(df, crop)
        sub = sub[sub['city_norm'] == city_norm]
        if sub.empty:
            out_rows.append({
                'requested_crop': crop,
                'matched_crop': None,
                'city': city,
                'year': None,
                'month': None,
                'min_price': None,
                'modal_price': None,
                'max_price': None,
                'note': 'No price found for this crop in the selected city.'
            })
            continue
        latest = sub.sort_values('__dt', ascending=False).iloc[0]
        out_rows.append({
            'requested_crop': crop,
            'matched_crop': latest.get('crop'),
            'city': latest.get('city'),
            'year': int(latest.get('year')) if pd.notna(latest.get('year')) else None,
            'month': int(latest.get('month')) if pd.notna(latest.get('month')) else None,
            'min_price': latest.get('min price') if 'min price' in sub.columns else latest.get('min_price'),
            'modal_price': latest.get('modal price') if 'modal price' in sub.columns else latest.get('modal_price'),
            'max_price': latest.get('max price') if 'max price' in sub.columns else latest.get('max_price'),
            'note': None
        })
    ranked = pd.DataFrame(out_rows)
    ranked['__sort'] = ranked['modal_price'].fillna(-1e18)
    ranked = ranked.sort_values('__sort', ascending=False).drop(columns='__sort')
    return ranked
