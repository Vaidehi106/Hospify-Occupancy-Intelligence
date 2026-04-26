# 🏨 Hospify — Occupancy Intelligence Platform

> A production-ready hotel revenue intelligence dashboard built with Streamlit and LightGBM. Forecast occupancy, analyse market segments, run rate what-if simulations, and plan staffing — all from a single dark-mode UI.

---

## Features

**Property Dashboard**
- 2026 YTD KPIs — Avg Occupancy, Total Rooms Sold, ADR, RevPAR, Total Revenue — all compared to prior year
- Year-over-Year trend charts (Revenue, Rooms Sold, ADR) with cumulative and running-average views
- Historical market segment contribution (horizontal bar + donut)
- Model validation panel — Actual vs Predicted occupancy, diverging error bars, room-error scatter with 7-day rolling average and tolerance band

**Occupancy Forecast**
- ML forecast up to 90 days ahead per property
- Seasonal intelligence narrative generated from forecast context
- Weekend highlighting, 80% occupancy target line
- ADR, RevPAR, and Revenue columns pulled from same-date last year

**Market Segments**
- EWMA pickup model harmonized to LightGBM room targets — segment rooms always sum exactly to total forecast rooms (no RoomsSold mismatch)
- Donut chart + scrollable glass detail table
- Dominant segment detection, avg harmonized occupancy KPI

**Rate What-If Simulator**
- Propose any rate change from −30% to +30%
- Per-property, per-day-of-week Price Elasticity of Demand (PED) engine — calibrated from historical ADR vs Occperc data
- 3-panel chart: occupancy comparison, DOW impact bar, daily revenue base vs adjusted
- DOW impact summary table with PED reference, base ADR, adjusted ADR, Δ occ pp, revenue delta
- Smart narrative: tells you whether the rate move is revenue-positive or negative

**Staff & Inventory**
- Housekeeping, Front Desk, F&B, Maintenance, Security headcount from forecast rooms
- Operational requirements from harmonized segment model (where available)
- Downloadable operations CSV

---

## Project Structure

```
hospify/
│
├── app.py                  # Streamlit entrypoint — UI, widgets, navigation
├── engine.py               # Forecast engine, MS harmonizer, PED what-if logic
├── components.py           # All Plotly chart builders and HTML helpers
├── theme.py                # Shared colour palette and Plotly layout template
├── style.css               # Dark-glass UI stylesheet
├── main.py                 # FastAPI backend (optional REST endpoint)
├── model_val_files.py      # Script to generate validation CSVs from actuals
├── requirements.txt
│
├── data/
│   ├── history_1.csv       # Aureon   — full occupancy history
│   ├── history_2.csv       # Velaris  — full occupancy history
│   ├── history_3.csv       # Zenvyra  — full occupancy history
│   ├── val_1.csv           # Aureon   — validation file (actual vs predicted)
│   ├── val_2.csv           # Velaris  — validation file
│   ├── val_3.csv           # Zenvyra  — validation file
│   ├── ms_forecast_1.csv   # Aureon   — EWMA segment forecast (StayDate + seg cols)
│   ├── ms_forecast_2.csv   # Velaris  — EWMA segment forecast
│   └── ms_forecast_3.csv   # Zenvyra  — EWMA segment forecast
│
├── models/
│   ├── data_1.pkl          # Aureon   — LightGBM ensemble package
│   ├── data_2.pkl          # Velaris  — LightGBM direct model
│   ├── data_3.pkl          # Zenvyra  — LightGBM direct model
│   ├── data1_MS.pkl        # Aureon   — EWMA MS model config
│   ├── data2_MS.pkl        # Velaris  — EWMA MS model config
│   └── data3_MS.pkl        # Zenvyra  — EWMA MS model config
│
└── assets/                 # Screenshots for README (optional)
```

---

## Properties & Model Config

| Property | Inventory | Engine    | Transform | Segments |
|----------|-----------|-----------|-----------|----------|
| Aureon   | 126 rooms | Ensemble  | None      | 13       |
| Velaris  | 100 rooms | Direct    | log1p⁻¹   | 11       |
| Zenvyra  | 185 rooms | Direct    | ∛ (cube)  | 9        |

**Engine types:**
- `ensemble` — 90% LightGBM (day-of-week models) + 10% year-naive blended forecast
- `direct` — single LightGBM model with lag and rolling-mean features, inverse-transformed on output

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/Vaidehi106/hospify.git
cd hospify
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Add your data & models

Place your files in `data/` and `models/` as shown in the project structure above.

History CSVs must have these columns:

```
Dates, DayOfWeek, Occperc, Inventory, ADR
```

MS forecast CSVs must have:

```
StayDate, [segment_col_1], [segment_col_2], ..., RoomsSold
```

Validation CSVs are generated automatically — see [Generating Validation Files](#generating-validation-files).

### 3. Run the dashboard

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### 4. (Optional) Run the FastAPI backend

```bash
uvicorn main:app --reload
```

API available at [http://localhost:8000/api/forecast?property=Aureon&horizon=30](http://localhost:8000)

---

## Generating Validation Files

Validation CSVs compare the model's predictions against real observed data. To regenerate them, edit the paths in `model_val_files.py` and run:

```bash
python model_val_files.py
```

The script calls `engine.py` directly, merges actuals with the forecast, computes MAE and room errors, and saves the result to `data/val_N.csv`.

---

## Data Flow

```
history_N.csv
      │
      ▼
 engine.py (LightGBM)
      │  Predicted_Occperc, Predicted_Rooms
      ▼
harmonize_forecasts()
      │  Merges with ms_forecast_N.csv (EWMA segment rooms)
      │  Scales segments so Σ(segments) = Target_Total_Rooms
      ▼
harmonized_df
      │
      ├──▶ Market Segments widget   (donut + table)
      ├──▶ Rate What-If Simulator   (PED-adjusted occ + revenue)
      └──▶ Staff & Inventory widget (housekeeping, front desk)
```

---

## Rate What-If — PED Logic

The simulator uses per-property, per-day-of-week **Price Elasticity of Demand (PED)** values hardcoded in `engine.py` under `_PED_BASE`. These were calibrated from the historical ADR vs occupancy relationship in each property's CSV.

```
ΔOcc% = PED × (ΔRate%) × Base_Occ
Adjusted_Occ = Base_Occ × (1 + PED × rate_change / 100)
Rev_Adjusted  = Adjusted_Rooms × Proposed_ADR
```

PED values are negative (inverse relationship). Weekends have lower elasticity (leisure, committed bookings). Weekdays are more elastic (corporate, price-sensitive transient).

| Property | Mon   | Fri   | Sat   |
|----------|-------|-------|-------|
| Aureon   | −0.55 | −0.30 | −0.25 |
| Velaris  | −0.70 | −0.28 | −0.22 |
| Zenvyra  | −0.38 | −0.18 | −0.15 |

To recalibrate, update `_PED_BASE`, `_BASE_ADR`, and `_BASE_OCC` in `engine.py`.

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `app.py` | All Streamlit UI — sidebar navigation, section rendering, widget logic |
| `engine.py` | `run_csv_forecast()`, `harmonize_forecasts()`, `apply_rate_whatif()`, all data loaders |
| `components.py` | Every chart function (`chart_forecast`, `chart_validation`, `chart_rate_whatif`, etc.) |
| `theme.py` | `PLOTLY_LAYOUT`, `COLORS`, `SEG_COLORS` — edit here to retheme globally |
| `style.css` | Dark glass UI — KPI cards, glass tables, page titles, insight cards |
| `main.py` | FastAPI `/api/forecast` endpoint (optional, for external integrations) |
| `model_val_files.py` | CLI script to regenerate `val_N.csv` from observed actuals |

---

## Tech Stack

| Layer | Library | Version |
|-------|---------|---------|
| UI | Streamlit | 1.56.0 |
| Charts | Plotly | 5.19.0 |
| ML | LightGBM | 4.6.0 |
| Data | Pandas / NumPy | 2.2.0 / 1.26.4 |
| API | FastAPI + Uvicorn | 0.135.2 / 0.42.0 |
| ML utils | scikit-learn | 1.8.0 |

Full dependency list: [`requirements.txt`](requirements.txt)

---

## Adding a New Property

1. Add history CSV to `data/` and models to `models/`
2. Add an entry to `PROPERTY_CFG` in `engine.py` following the existing pattern
3. Add the property name to the `st.selectbox` options in `app.py`
4. Add PED values to `_PED_BASE`, `_BASE_ADR`, `_BASE_OCC` in `engine.py`
5. (Optional) Add historical segment CSV path to `_MS_HISTORY_CSV` in `app.py`

---


## Author

Vaidehi Patel — Innovative Machine Learning Engineer
For questions or contributions, open an issue or pull request.
