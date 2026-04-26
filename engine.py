"""
engine.py — CSV-based forecast engine + MS segment harmonizer
Uses frozen history CSVs + pre-trained pkl models.

History CSV columns expected:  Dates, DayOfWeek, Occperc, Inventory, ADR
Transform types per property:
  Aureon:  none   → raw output
  Velaris: log1p  → inverse: expm1(raw)
  Zenvyra: cube   → inverse: cbrt(raw)

MS pkl files store:  { 'model_type': 'EWMA_Pickup', 'best_alpha': float, 'inventory_cap': int }
MS forecast CSVs:    StayDate, [seg_col_1 ... seg_col_N], RoomsSold
                     (pre-computed by running the notebooks and saving output)
"""

import os
import re
import math
import pandas as pd
import numpy as np
import pickle
from datetime import timedelta
import streamlit as st

_model_cache: dict = {}

DATA_DIR  = "data"
MODEL_DIR = "models"

# ─────────────────────────────────────────────────────────────────────────────
# PROPERTY CONFIG  (2 new keys per property: ms_model_path, ms_forecast_csv)
# ─────────────────────────────────────────────────────────────────────────────
PROPERTY_CFG = {
    "Aureon": {
        # ── existing keys ──────────────────────────────────────────────────
        "history_csv":    os.path.join(DATA_DIR, "history_1.csv"),
        "val_csv":        os.path.join(DATA_DIR, "val_1.csv"),
        "dow_csv":        os.path.join(DATA_DIR, "dow_avg_1.csv"),
        "model_key":      "data_1",
        "model_path":     os.path.join(MODEL_DIR, "data_1.pkl"),
        "fetch_limit":    400,
        "engine_type":    "ensemble",
        "transform":      "none",
        "jan_correction": False,
        "inventory":      126,
        # ── NEW: MS segment model ──────────────────────────────────────────
        "ms_model_path":   os.path.join(MODEL_DIR, "data1_MS.pkl"),
        "ms_forecast_csv": os.path.join(DATA_DIR, "ms_forecast_1.csv"),
        "ms_segments": [
            "BRAND DISCOUNT", "OTA DISCOUNT", "CORPORATE NEGOTIATED", "RETAIL",
            "LOCAL NEGOTIATED", "OPAQUE & PACKAGE", "QUALIFIED DISCOUNT",
            "CORPORATE DISCOUNT", "EMPLOYEE", "CONSORTIA", "REDEMPTION",
            "GOVERNMENT", "OTA RETAIL",
        ],
    },
    "Velaris": {
        # ── existing keys ──────────────────────────────────────────────────
        "history_csv":    os.path.join(DATA_DIR, "history_2.csv"),
        "val_csv":        os.path.join(DATA_DIR, "val_2.csv"),
        "model_key":      "data_2",
        "model_path":     os.path.join(MODEL_DIR, "data_2.pkl"),
        "fetch_limit":    60,
        "engine_type":    "direct",
        "transform":      "log1p",
        "jan_correction": True,
        "inventory":      100,
        # ── NEW: MS segment model ──────────────────────────────────────────
        "ms_model_path":   os.path.join(MODEL_DIR, "data2_MS.pkl"),
        "ms_forecast_csv": os.path.join(DATA_DIR, "ms_forecast_2.csv"),
        "ms_segments": [
            "RETAIL", "OTA DISCOUNT", "BRAND DISCOUNT", "QUALIFIED DISCOUNT",
            "CORPORATE NEGOTIATED", "OTA RETAIL", "LOCAL NEGOTIATED",
            "CORPORATE DISCOUNT", "OPAQUE & PACKAGE", "EMPLOYEE", "GOVERNMENT",
        ],
    },
    "Zenvyra": {
        # ── existing keys ──────────────────────────────────────────────────
        "history_csv":    os.path.join(DATA_DIR, "history_3.csv"),
        "val_csv":        os.path.join(DATA_DIR, "val_3.csv"),
        "model_key":      "data_3",
        "model_path":     os.path.join(MODEL_DIR, "data_3.pkl"),
        "fetch_limit":    60,
        "engine_type":    "direct",
        "transform":      "cube",
        "jan_correction": False,
        "inventory":      185,
        # ── NEW: MS segment model ──────────────────────────────────────────
        "ms_model_path":   os.path.join(MODEL_DIR, "data3_MS.pkl"),
        "ms_forecast_csv": os.path.join(DATA_DIR, "ms_forecast_3.csv"),
        "ms_segments": [
            "DIST", "BAR", "CONS", "CORP", "HOUSE", "PKG", "SPEC", "WHOLE", "COMP",
        ],
    },
}


# ─── Config helpers ───────────────────────────────────────────────────────────
def _get_base(property_choice: str) -> str:
    return property_choice.split(":")[0].strip()

def _get_cfg(property_choice: str) -> dict:
    base = _get_base(property_choice)
    if base not in PROPERTY_CFG:
        raise ValueError(f"Unknown property: {property_choice}")
    return PROPERTY_CFG[base]

def _get_model(key: str, path: str):
    if key not in _model_cache:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")
        with open(path, "rb") as f:
            _model_cache[key] = pickle.load(f)
        print(f"Model loaded: {key}")
    return _model_cache[key]


# ─── Data loading ─────────────────────────────────────────────────────────────
@st.cache_data
def fetch_forecast_data(property_choice: str) -> pd.DataFrame:
    cfg = _get_cfg(property_choice)
    df  = pd.read_csv(cfg["history_csv"])
    df["Dates"] = pd.to_datetime(df["Dates"])
    df = df.sort_values("Dates").reset_index(drop=True)
    return df.tail(cfg["fetch_limit"]).reset_index(drop=True)

def fetch_yoy_data(property_choice: str, forecast_month: int) -> pd.DataFrame:
    cfg = _get_cfg(property_choice)
    df  = pd.read_csv(cfg["history_csv"])
    df["Dates"] = pd.to_datetime(df["Dates"])
    return df[df["Dates"].dt.month == forecast_month].copy()

def load_val_csv(property_choice: str) -> pd.DataFrame:
    cfg = _get_cfg(property_choice)
    if not os.path.exists(cfg["val_csv"]):
        return pd.DataFrame()
    df = pd.read_csv(cfg["val_csv"])
    df["Dates"] = pd.to_datetime(df["Dates"])
    return df

def load_full_history(property_choice: str) -> pd.DataFrame:
    cfg = _get_cfg(property_choice)
    df  = pd.read_csv(cfg["history_csv"])
    df["Dates"] = pd.to_datetime(df["Dates"])
    return df.sort_values("Dates").reset_index(drop=True)

def compute_dow_month_avg(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["dow_num"] = tmp["Dates"].dt.dayofweek
    tmp["Month"]   = tmp["Dates"].dt.month
    return (
        tmp.groupby(["dow_num", "Month"])["Occperc"]
        .mean().reset_index().rename(columns={"Occperc": "DowMonth_Avg"})
    )


# ─── ADR lookup from history ──────────────────────────────────────────────────
def fetch_adr_for_forecast(property_choice: str, forecast_dates) -> list:
    cfg = _get_cfg(property_choice)
    df  = pd.read_csv(cfg["history_csv"])
    df["Dates"] = pd.to_datetime(df["Dates"])

    if "ADR" not in df.columns:
        return [0.0] * len(forecast_dates)

    result = []
    for d in pd.to_datetime(forecast_dates):
        ly_date = d - pd.DateOffset(years=1)
        match   = df[df["Dates"] == ly_date]
        if not match.empty:
            result.append(float(match["ADR"].iloc[0]))
        else:
            month_vals = df[df["Dates"].dt.month == d.month]["ADR"].dropna()
            result.append(float(month_vals.mean()) if not month_vals.empty else 0.0)
    return result


# ─── Inverse transform helper ─────────────────────────────────────────────────
def _apply_inverse_transform(raw: float, transform: str) -> float:
    if transform == "log1p":
        return float(np.clip(np.expm1(raw), 0, 100))
    elif transform == "cube":
        return float(np.clip(np.cbrt(raw), 0, 100))
    else:
        return float(np.clip(raw, 0, 100))


# ─── Ensemble engine (Aureon) ─────────────────────────────────────────────────
def _forecast_ensemble(dow_models, df, dow_month_avg_df, feature_cols, horizon_days):
    history        = df.copy()
    actual_history = df.copy()
    last_date      = history["Dates"].max()
    weights        = {"lgbm": 0.90, "year_naive": 0.10}
    results        = []

    for i in range(1, int(horizon_days) + 1):
        next_date   = last_date + pd.Timedelta(days=i)
        current_dow = int(next_date.dayofweek)
        occ_series  = history["Occperc"]
        occ_actual  = actual_history["Occperc"]
        forecast_inv = history.get("Inventory", pd.Series([100] * len(history))).iloc[-1]

        row = {
            "dow_num": current_dow, "Month": next_date.month,
            "Quarter": (next_date.month - 1) // 3 + 1, "DayOfMonth": next_date.day,
            "IsMonthStart": int(next_date.day <= 7), "IsMonthEnd": int(next_date.day >= 24),
        }
        for lag in [21, 28, 364]:
            idx = len(occ_series) - lag
            row[f"Occ_Lag_{lag}d"] = occ_series.iloc[idx] if idx >= 0 else occ_series.mean()

        ya = len(occ_actual) - 364
        row["Occ_YearAgo_RollMean"] = (
            occ_actual.iloc[max(0, ya - 3): ya + 4].mean() if ya > 0 else occ_actual.mean()
        )
        row["Occ_RollMean_56d"]  = occ_series.tail(56).mean()
        row["Occ_Trend_14d"]     = occ_series.tail(14).mean() - occ_series.iloc[-28:-14].mean()
        row["Occ_Trend_28d"]     = occ_series.tail(28).mean() - occ_series.iloc[-56:-28].mean()
        last_8  = occ_series.iloc[-8]  if len(occ_series) >= 8  else occ_series.iloc[-1]
        last_29 = occ_series.iloc[-29] if len(occ_series) >= 29 else occ_series.iloc[-1]
        row["Occ_PctChg_7d"]     = np.clip((occ_series.iloc[-1] / last_8  - 1) if last_8  != 0 else 0, -1, 1)
        row["Occ_PctChg_28d"]    = np.clip((occ_series.iloc[-1] / last_29 - 1) if last_29 != 0 else 0, -1, 1)
        row["Momentum_Velocity"] = row["Occ_PctChg_7d"] - row["Occ_PctChg_28d"]

        lookup = dow_month_avg_df[
            (dow_month_avg_df["dow_num"] == current_dow) &
            (dow_month_avg_df["Month"] == next_date.month)
        ]
        dow_avg = float(lookup["DowMonth_Avg"].values[0]) if len(lookup) > 0 else occ_series.mean()
        row["DowMonth_Avg"] = dow_avg

        if current_dow in dow_models:
            lgbm_pred = float(np.clip(
                dow_models[current_dow].predict(pd.DataFrame([row])[feature_cols])[0], 0, 100
            ))
        else:
            lgbm_pred = dow_avg

        pred = float(np.clip(
            weights["lgbm"] * lgbm_pred + weights["year_naive"] * row.get("Occ_Lag_364d", lgbm_pred),
            0, 100
        ))
        results.append({
            "Date": next_date.strftime("%Y-%m-%d"), "DayOfWeek": next_date.strftime("%A"),
            "Predicted_Occperc": round(pred, 2), "Inventory": int(forecast_inv),
            "Predicted_Rooms": round(pred * forecast_inv / 100, 1),
        })
        history = pd.concat([history, pd.DataFrame([{
            "Dates": next_date, "Occperc": pred, "Inventory": forecast_inv,
            "dow_num": current_dow, "Month": next_date.month
        }])], ignore_index=True)

    return pd.DataFrame(results)


def _model_feature_names(model) -> list:
    if hasattr(model, 'feature_name_'):
        return list(model.feature_name_)
    elif hasattr(model, 'booster_') and hasattr(model.booster_, 'feature_name'):
        return list(model.booster_.feature_name())
    return []


# ─── Direct engine (Velaris, Zenvyra) ──────────────────────────────────────────
def _forecast_direct(df, horizon_days, model, transform: str = "none", jan_correction: bool = False):
    df = df.copy()
    df["Dates"] = pd.to_datetime(df["Dates"], errors="coerce")
    df = df.sort_values("Dates").reset_index(drop=True)

    inventory      = float(df["Inventory"].iloc[-1]) if "Inventory" in df.columns else 100.0
    last_date      = df["Dates"].iloc[-1]
    expected       = _model_feature_names(model)
    lag_cols       = sorted([int(re.search(r"\d+", c).group()) for c in expected if c.startswith("lag_")])
    rolling_cols   = sorted([int(re.search(r"\d+", c).group()) for c in expected if c.startswith("rolling_mean_")])
    occ_history    = list(df["Occperc"].values)
    actual_history = list(df["Occperc"].values)

    def _fuzzy(h): return float(np.mean(h[-7:] if len(h) >= 7 else h))

    future_preds = []
    for step in range(1, int(horizon_days) + 1):
        next_date = last_date + timedelta(days=step)
        row = {}
        time_feats = {
            "week_of_year":  int(next_date.isocalendar()[1]),
            "DayOfWeek_Num": int(next_date.dayofweek),
            "DayOfWeek":     int(next_date.dayofweek),
            "month":         int(next_date.month),
            "is_holiday":    int(next_date.dayofweek in [4, 5]),
            "day_of_month":  int(next_date.day),
            "quarter":       int((next_date.month - 1) // 3 + 1),
            "day_of_year":   int(next_date.timetuple().tm_yday),
            "WeekOfMonth":   int((next_date.day - 1) // 7 + 1),
            "Inventory":     inventory,
            "Fuzzy_Score":   _fuzzy(actual_history),
        }
        for k, v in time_feats.items():
            if k in expected:
                row[k] = v
        for lag in lag_cols:
            idx = len(occ_history) - lag
            row[f"lag_{lag}"] = float(occ_history[idx]) if idx >= 0 else float(occ_history[0])
        for window in rolling_cols:
            shift = 7; s = len(actual_history) - shift - window; e = len(actual_history) - shift
            vals = actual_history[max(0, s): e]
            row[f"rolling_mean_{window}"] = float(np.mean(vals)) if vals else float(actual_history[-1])
        for feat in expected:
            if feat not in row:
                row[feat] = 0.0

        input_data = pd.DataFrame([row])[expected] if expected else pd.DataFrame([row])
        raw        = float(model.predict(input_data)[0])
        pred_val   = _apply_inverse_transform(raw, transform)

        future_preds.append({
            "Date":              pd.to_datetime(next_date).strftime("%Y-%m-%d"),
            "DayOfWeek":         pd.to_datetime(next_date).strftime("%A"),
            "Predicted_Occperc": round(pred_val, 2),
            "Inventory":         inventory,
            "Predicted_Rooms":   round(pred_val * inventory / 100, 1),
        })
        occ_history.append(pred_val)

    df_fc = pd.DataFrame(future_preds)
    df_fc["Date"] = pd.to_datetime(df_fc["Date"])

    if jan_correction:
        jan = (df_fc["Date"].dt.month == 1) & (df_fc["Date"].dt.day <= 15)
        we  = df_fc["Date"].dt.dayofweek.isin([4, 5])
        df_fc.loc[jan & we,  "Predicted_Occperc"] *= 0.75
        df_fc.loc[jan & ~we, "Predicted_Occperc"] *= 0.35

    df_fc["Predicted_Occperc"] = np.clip(df_fc["Predicted_Occperc"], 0, 100)
    df_fc["Predicted_Rooms"]   = (df_fc["Predicted_Occperc"] * inventory / 100).round(1)
    return df_fc


# ─── Main forecast entry point ────────────────────────────────────────────────
def run_csv_forecast(property_choice: str, horizon) -> pd.DataFrame:
    cfg  = _get_cfg(property_choice)
    base = _get_base(property_choice)
    try:
        df = fetch_forecast_data(property_choice)

        if cfg["engine_type"] == "ensemble":
            pkg   = _get_model(cfg["model_key"], cfg["model_path"])
            df_fc = _forecast_ensemble(
                pkg["models"], df, compute_dow_month_avg(df), pkg["feature_columns"], horizon
            )
        else:
            model = _get_model(cfg["model_key"], cfg["model_path"])
            df_fc = _forecast_direct(
                df, horizon, model,
                transform=cfg.get("transform", "none"),
                jan_correction=cfg.get("jan_correction", False),
            )

        if df_fc.empty:
            return df_fc

        df_fc["Date"] = pd.to_datetime(df_fc["Date"])
        df_fc["ADR"]       = fetch_adr_for_forecast(property_choice, df_fc["Date"])
        df_fc["RoomsSold"] = (df_fc["Predicted_Occperc"] / 100 * df_fc["Inventory"]).round(0).astype(int)
        df_fc["Revenue"]   = (df_fc["ADR"] * df_fc["RoomsSold"]).round(2)
        df_fc["RevPAR"]    = (df_fc["ADR"] * df_fc["Predicted_Occperc"] / 100).round(2)

        return df_fc

    except Exception as e:
        import traceback
        print(f"CSV Engine Error [{base}]: {e}")
        traceback.print_exc()
        return pd.DataFrame()


# ─── Aliases ─────────────────────────────────────────────────────────────────
run_forecast = run_csv_forecast
load_val     = load_val_csv
get_yoy_data = fetch_yoy_data


# ═══════════════════════════════════════════════════════════════════════════════
# MS SEGMENT LOADER
# ═══════════════════════════════════════════════════════════════════════════════
def load_ms_forecast(property_choice: str) -> pd.DataFrame:
    """
    Load the pre-computed MS segment forecast CSV for this property.
    Returns empty DataFrame if file not found (graceful degradation).

    Expected CSV columns: StayDate, [segment cols...], RoomsSold
    Generated by running your notebooks and saving final_report.to_csv(...)
    """
    cfg      = _get_cfg(property_choice)
    csv_path = cfg.get("ms_forecast_csv", "")

    if not csv_path or not os.path.exists(csv_path):
        return pd.DataFrame()

    try:
        df = pd.read_csv(csv_path)
        df["StayDate"] = pd.to_datetime(df["StayDate"])
        return df
    except Exception as e:
        print(f"MS forecast load error [{property_choice}]: {e}")
        return pd.DataFrame()


def load_ms_config(property_choice: str) -> dict:
    """
    Load the MS pkl config (best_alpha, inventory_cap, model_type).
    Returns empty dict if file not found.
    """
    cfg      = _get_cfg(property_choice)
    pkl_path = cfg.get("ms_model_path", "")

    if not pkl_path or not os.path.exists(pkl_path):
        return {}

    try:
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"MS pkl load error [{property_choice}]: {e}")
        return {}


# ═══════════════════════════════════════════════════════════════════════════════
# HARMONIZER  — the bridge between LightGBM and EWMA segment model
# ═══════════════════════════════════════════════════════════════════════════════
def harmonize_forecasts(
    lightgbm_df: pd.DataFrame,
    ms_forecast_df: pd.DataFrame,
    inventory: int,
    property_choice: str,
) -> pd.DataFrame:
    """
    Reconciles LightGBM occupancy % with EWMA segment room counts.

    Args:
        lightgbm_df:    Must have columns: Date (datetime), Predicted_Occperc
        ms_forecast_df: Must have columns: StayDate (datetime), [seg cols], RoomsSold
        inventory:      Total rooms (integer)
        property_choice: e.g. "Aureon"

    Returns:
        DataFrame with: Date, DayOfWeek, Target_Total_Rooms, Predicted_Occperc,
                        [harmonized segment cols], ADR, Revenue, RevPAR (if present)
    """
    cfg      = _get_cfg(property_choice)
    segments = cfg.get("ms_segments", [])

    # 1. Compute authoritative room target from LightGBM
    lgbm = lightgbm_df.copy()
    lgbm["Date"] = pd.to_datetime(lgbm["Date"])
    lgbm["Target_Total_Rooms"] = (lgbm["Predicted_Occperc"] / 100 * inventory).round().astype(int)

    if ms_forecast_df.empty or not segments:
        # No MS data — return LightGBM frame with zero-filled segments
        lgbm["RoomsSold"] = lgbm["Target_Total_Rooms"]
        for seg in segments:
            lgbm[seg] = 0
        return lgbm

    # 2. Align MS forecast to forecast date range
    ms = ms_forecast_df.copy()
    ms["StayDate"] = pd.to_datetime(ms["StayDate"])

    # Keep only segment columns that actually exist in the CSV
    present_segs = [s for s in segments if s in ms.columns]
    missing_segs = [s for s in segments if s not in ms.columns]

    # 3. Merge on date
    merged = pd.merge(
        lgbm,
        ms[["StayDate"] + present_segs + (["RoomsSold"] if "RoomsSold" in ms.columns else [])],
        left_on="Date", right_on="StayDate",
        how="left",
    )
    merged.drop(columns=["StayDate"], errors="ignore", inplace=True)

    # Fill missing dates (MS might not cover all forecast dates) with equal split
    n_segs = max(len(present_segs), 1)
    for seg in present_segs:
        merged[seg] = merged[seg].fillna(merged["Target_Total_Rooms"] / n_segs)
    if "RoomsSold" not in merged.columns:
        merged["RoomsSold"] = merged["Target_Total_Rooms"]
    else:
        merged["RoomsSold"] = merged["RoomsSold"].fillna(merged["Target_Total_Rooms"])

    # 4. Compute proportional ratios from EWMA RoomsSold
    #    ratio_i = seg_i / RoomsSold  (handle zero division → equal share)
    for seg in present_segs:
        ratio_col = f"_ratio_{seg}"
        merged[ratio_col] = np.where(
            merged["RoomsSold"] > 0,
            merged[seg] / merged["RoomsSold"],
            1.0 / n_segs,
        )

    # 5. Multiply Target_Total_Rooms × ratio → harmonized segment rooms
    for seg in present_segs:
        ratio_col = f"_ratio_{seg}"
        merged[seg] = (merged["Target_Total_Rooms"] * merged[ratio_col]).round().astype(int)
        merged.drop(columns=[ratio_col], inplace=True)

    # 6. Fill any segments missing from MS CSV with zero
    for seg in missing_segs:
        merged[seg] = 0

    # 7. Recompute RoomsSold from harmonized segments (sum of all segs) — single source of truth
    merged["RoomsSold"] = merged[present_segs].sum(axis=1).clip(upper=inventory)

    # Drop any merge artefact columns (_x, _y suffixes)
    for col in list(merged.columns):
        if col.endswith("_x") or col.endswith("_y"):
            merged.drop(columns=[col], inplace=True, errors="ignore")

    return merged


# ═══════════════════════════════════════════════════════════════════════════════
# OPERATIONAL METRICS
# ═══════════════════════════════════════════════════════════════════════════════
def calculate_operations(harmonized_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds operational staffing / linen columns to the harmonized DataFrame.

    New columns added:
        Required_Housekeepers  — ceil(Target_Total_Rooms / 15)
        Front_Desk_Agents      — max(1, ceil(Target_Total_Rooms / 30))
        Linen_Sets_Required    — Target_Total_Rooms × 2
    """
    df = harmonized_df.copy()
    rooms = df["Target_Total_Rooms"]

    df["Required_Housekeepers"] = np.ceil(rooms / 15).astype(int)
    df["Front_Desk_Agents"]     = np.maximum(1, np.ceil(rooms / 30)).astype(int)
    df["Linen_Sets_Required"]   = (rooms * 2).astype(int)

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# EXISTING HELPERS (kept unchanged for backward compatibility)
# ═══════════════════════════════════════════════════════════════════════════════

# Static segment base — used by dashboard "segments" widget only
# (real forecasts now use harmonize_forecasts + MS CSV instead)
_SEGMENT_BASE = {
    "Aureon":  {"Business": 45, "Leisure": 30, "Group": 15, "OTA / Online": 10},
    "Velaris": {"Business": 20, "Leisure": 60, "Group":  5, "OTA / Online": 15},
    "Zenvyra": {"Business": 60, "Leisure": 10, "Group": 25, "OTA / Online":  5},
}

def build_segment_forecast(df: pd.DataFrame, property_choice: str) -> pd.DataFrame:
    """Legacy helper — used in dashboard segment widget (static ratios)."""
    seg_df = df.copy()
    base   = property_choice.split(":")[0].strip()
    mix    = _SEGMENT_BASE.get(base, _SEGMENT_BASE["Aureon"])
    for segment, pct in mix.items():
        seg_df[segment] = seg_df["Predicted_Occperc"] * (pct / 100.0)
    return seg_df

def build_staff_plan(df: pd.DataFrame) -> pd.DataFrame:
    """Legacy helper — used in Staff & Inventory widget."""
    staff_df = df.copy()
    rooms = staff_df["Predicted_Rooms"]
    staff_df["Housekeeping"]     = np.ceil(rooms / 15)
    staff_df["Front Desk"]       = np.ceil(rooms / 50).clip(lower=1)
    staff_df["F&B / Restaurant"] = np.ceil(rooms / 40)
    staff_df["Maintenance"]      = np.ceil(rooms / 80).clip(lower=1)
    staff_df["Security"]         = np.ceil(rooms / 100).clip(lower=1)
    staff_df["Total Staff"]      = staff_df[["Housekeeping", "Front Desk", "F&B / Restaurant",
                                             "Maintenance", "Security"]].sum(axis=1)
    staff_df.rename(columns={"Predicted_Occperc": "Occupancy %"}, inplace=True)
    return staff_df


# ═══════════════════════════════════════════════════════════════════════════════
# PRICE ELASTICITY (PED) WHAT-IF SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════════

# Base PED (Price Elasticity of Demand) per property per day-of-week
# Derived from historical ADR vs Occperc patterns.
# Values represent % change in occupancy per 1% change in rate (negative = inverse).
# Weekends are less elastic (leisure, committed bookings).
# Weekdays more elastic (corporate, price-sensitive transient).
_PED_BASE = {
    "Aureon": {
        # inventory=126, avg ADR ~$103, mid-range business+leisure mix
        "Mon": -0.55,
        "Tue": -0.52,
        "Wed": -0.50,
        "Thu": -0.48,
        "Fri": -0.30,
        "Sat": -0.25,
        "Sun": -0.60,
    },
    "Velaris": {
        # inventory=100, weekend-heavy leisure, very rate-sensitive midweek
        "Mon": -0.70,
        "Tue": -0.68,
        "Wed": -0.65,
        "Thu": -0.60,
        "Fri": -0.28,
        "Sat": -0.22,
        "Sun": -0.72,
    },
    "Zenvyra": {
        # inventory=185, high ADR luxury, less elastic overall
        "Mon": -0.38,
        "Tue": -0.35,
        "Wed": -0.33,
        "Thu": -0.32,
        "Fri": -0.18,
        "Sat": -0.15,
        "Sun": -0.40,
    },
}

# Base (historical median) ADR per property per day-of-week (computed from history CSVs)
_BASE_ADR = {
    "Aureon": {
        "Mon": 101.6, "Tue": 103.2, "Wed": 103.0, "Thu": 102.4,
        "Fri": 105.1, "Sat": 106.1, "Sun": 99.8,
    },
    "Velaris": {
        "Mon": 69.3, "Tue": 74.6, "Wed": 74.6, "Thu": 76.6,
        "Fri": 112.9, "Sat": 117.3, "Sun": 75.8,
    },
    "Zenvyra": {
        "Mon": 524.7, "Tue": 574.3, "Wed": 572.0, "Thu": 534.5,
        "Fri": 564.6, "Sat": 573.5, "Sun": 463.5,
    },
}

# Base (historical median) Occperc per property per day-of-week
_BASE_OCC = {
    "Aureon": {
        "Mon": 55.2, "Tue": 63.8, "Wed": 63.2, "Thu": 59.3,
        "Fri": 70.9, "Sat": 73.8, "Sun": 44.5,
    },
    "Velaris": {
        "Mon": 28.3, "Tue": 31.5, "Wed": 32.0, "Thu": 37.1,
        "Fri": 50.8, "Sat": 54.5, "Sun": 21.5,
    },
    "Zenvyra": {
        "Mon": 77.6, "Tue": 83.9, "Wed": 81.9, "Thu": 79.2,
        "Fri": 85.1, "Sat": 88.1, "Sun": 69.7,
    },
}


def apply_rate_whatif(
    forecast_df: pd.DataFrame,
    property_choice: str,
    rate_change_pct: float,
    rooms_out: int = 0,
) -> pd.DataFrame:
    """
    Apply a proposed ADR rate change (%) to the forecast and compute
    the PED-adjusted occupancy impact per day-of-week.

    Args:
        forecast_df:      LightGBM forecast DataFrame (must have Date, DayOfWeek,
                          Predicted_Occperc, Inventory, ADR columns)
        property_choice:  "Aureon" / "Velaris" / "Zenvyra"
        rate_change_pct:  Proposed rate change as % of base (e.g. +10 = raise 10%)
        rooms_out:        Rooms offline (renovation etc.)

    Returns:
        DataFrame with new columns:
            Base_ADR, Proposed_ADR, PED, Base_Occperc,
            Adjusted_Occperc, Adjusted_Rooms, Occ_Delta_pp,
            Rev_Base, Rev_Adjusted, Rev_Delta
    """
    base   = property_choice.split(":")[0].strip()
    ped_map = _PED_BASE.get(base, _PED_BASE["Aureon"])
    adr_map = _BASE_ADR.get(base, _BASE_ADR["Aureon"])
    inventory = int(forecast_df["Inventory"].iloc[0])
    eff_inv   = max(1, inventory - rooms_out)

    adj = forecast_df.copy()

    # Resolve 3-letter DOW abbreviation from full day name
    dow_abbr = adj["DayOfWeek"].str[:3]

    adj["Base_ADR"]      = dow_abbr.map(adr_map).fillna(adj.get("ADR", 0))
    adj["Proposed_ADR"]  = (adj["Base_ADR"] * (1 + rate_change_pct / 100)).round(2)
    adj["PED"]           = dow_abbr.map(ped_map).fillna(-0.50)
    adj["Base_Occperc"]  = adj["Predicted_Occperc"]

    # PED formula: ΔOcc% = PED × (ΔRate%) × Base_Occ
    # We use proportional form: new_occ = base_occ × (1 + PED × rate_change_pct/100)
    adj["Adjusted_Occperc"] = (
        adj["Base_Occperc"] * (1 + adj["PED"] * rate_change_pct / 100)
    ).clip(0, 100).round(2)

    adj["Adjusted_Rooms"] = (adj["Adjusted_Occperc"] / 100 * eff_inv).round(0).astype(int)
    adj["Occ_Delta_pp"]   = (adj["Adjusted_Occperc"] - adj["Base_Occperc"]).round(2)

    # Revenue impact
    base_rooms            = (adj["Base_Occperc"] / 100 * eff_inv).round(0)
    adj["Rev_Base"]       = (base_rooms * adj["Base_ADR"]).round(0)
    adj["Rev_Adjusted"]   = (adj["Adjusted_Rooms"] * adj["Proposed_ADR"]).round(0)
    adj["Rev_Delta"]      = (adj["Rev_Adjusted"] - adj["Rev_Base"]).round(0)

    return adj
