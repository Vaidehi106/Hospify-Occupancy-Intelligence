# app.py
# Run with: streamlit run app.py

import os
import calendar
import base64
import numpy as np
import pandas as pd
import streamlit as st

from engine import (
    run_csv_forecast,
    load_val_csv,
    fetch_yoy_data,
    build_segment_forecast,
    build_staff_plan,
    PROPERTY_CFG,
    load_full_history,
    load_ms_forecast,
    load_ms_config,
    harmonize_forecasts,
    calculate_operations,
    apply_rate_whatif,
)

from components import (
    season_insight,
    kpi_card,
    section_header,
    render_glass_table,
    chart_forecast,
    chart_validation,
    chart_yoy,
    chart_yoy_revenue,
    build_yoy_table,
    chart_segment_history_bar,
    chart_segment_donut,
    chart_segment_heatmap,
    chart_whatif,
    chart_whatif_horizontal_100pct,
    chart_operations,
    chart_staff,
    chart_rate_whatif,
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hospify — Occupancy Intelligence",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# CSS & BACKGROUND
# ─────────────────────────────────────────────────────────────────────────────
def _load_css(file_name: str) -> None:
    try:
        with open(file_name, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

_load_css("style.css")


def _set_background(image_file: str) -> None:
    try:
        with open(image_file, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        st.markdown(f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(10,25,47,0.82), rgba(10,25,47,0.82)),
                              url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>""", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

_set_background("image_1.png")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _engine_badge(cfg: dict) -> str:
    e = cfg.get("engine_type", "direct")
    t = cfg.get("transform", "none")
    label = "Ensemble" if e == "ensemble" else "Direct"
    transform_map = {"none": "—", "log1p": "log1p⁻¹", "cube": "∛"}
    return f"{label} · {transform_map.get(t, t)}"


def _history_date_range(property_choice: str) -> str:
    try:
        df = load_full_history(property_choice)
        return (
            f"{df['Dates'].min().strftime('%b %Y')} → "
            f"{df['Dates'].max().strftime('%b %Y')} ({len(df)} days)"
        )
    except Exception:
        return "N/A"


def _delta_color(val: float) -> str:
    return "🟢" if val >= 0 else "🔴"


def _safe_mean(s: pd.Series) -> float:
    return round(float(s.mean()), 2) if not s.empty else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# MS CSV PATH MAPPING  (historical segment data for dashboard bar chart)
# Maps property name → the CSV with StayDate / MarketSegment / RoomNight columns
# ─────────────────────────────────────────────────────────────────────────────
_MS_HISTORY_CSV = {
    "Aureon":  os.path.join("data", "MarketSegment_historical_data_aureon.csv"),
    "Velaris": os.path.join("data", "MarketSegment_historical_data_velaris.csv"),
    "Zenvyra": os.path.join("data", "MarketSegment_historical_data_zenvyra.csv"),
}


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────────────────────
for _k, _v in [
    ("active_section", None),
    ("active_widget",  None),
    ("harmonized_df",  None),
    ("ops_df",         None),
    ("ms_loaded",      False),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # ── Brand header (NO hotel image — removed as requested) ──────────────────
    st.markdown("""
    <div style="margin-bottom:1.25rem;">
      <div style="font-size:1.6rem;font-weight:800;
                  background:linear-gradient(135deg,#17C3B2,#60A5FA);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        HOSPIFY
      </div>
      <div style="font-size:0.72rem;color:#8892B0;letter-spacing:0.14em;text-transform:uppercase;">
        Occupancy Intelligence Platform
      </div>
    </div>
    """, unsafe_allow_html=True)

    property_choice = st.selectbox(
        "🏨 Property",
        options=["Aureon", "Velaris", "Zenvyra"],
    )

    base_key = property_choice.strip()
    cfg      = PROPERTY_CFG.get(base_key, {})
    inv      = cfg.get("inventory", "—")
    badge    = _engine_badge(cfg)

    ms_csv_path   = cfg.get("ms_forecast_csv", "")
    ms_available  = os.path.exists(ms_csv_path) if ms_csv_path else False
    ms_badge       = "✅ MS Model Ready" if ms_available else "⚠️ MS CSV not found"
    ms_badge_color = "#17C3B2" if ms_available else "#F4A261"

    st.markdown(f"""
    <div style="background:rgba(23,195,178,0.07);border:1px solid rgba(23,195,178,0.2);
                border-radius:10px;padding:0.75rem 1rem;margin:0.5rem 0 1rem 0;font-size:0.8rem;">
      <div style="color:#8892B0;text-transform:uppercase;letter-spacing:0.07em;
                  font-size:0.68rem;margin-bottom:0.5rem;">Property Details</div>
      <div style="color:#CCD6F6;margin-bottom:0.25rem;">🔑 <b>{inv}</b> rooms total</div>
      <div style="color:#CCD6F6;margin-bottom:0.25rem;">⚙️  {badge}</div>
      <div style="color:{ms_badge_color};margin-bottom:0.25rem;">📊 {ms_badge}</div>
    </div>
    """, unsafe_allow_html=True)

    horizon = st.radio(
        "📅 Forecast Horizon (days)",
        options=["7", "14", "30", "90"],
        index=2,
        horizontal=True,
    )

    st.markdown("<div style='margin:0.75rem 0;'></div>", unsafe_allow_html=True)

    # ── Navigation buttons ────────────────────────────────────────────────────
    dash_is_active = st.session_state.get("active_section") == "dashboard"
    fc_is_active   = st.session_state.get("active_section") == "forecast"

    dash_btn = st.button(
        "🏨 Property Dashboard",
        width="stretch",
        help="Load 2026 YTD KPIs and historical analysis widgets",
    )

    if dash_is_active:
        st.markdown("""
        <div style="margin:0.3rem 0 0.2rem 0.5rem;padding:0.7rem 0.9rem;
                    background:rgba(23,195,178,0.07);">
        """, unsafe_allow_html=True)

        dash_widgets = {
            "yoy_trend":  "📅  YOY Trend",
            "segments":   "🏷  Market Segments",
            "validation": "✅  Model Validation",
        }
        for wkey, wlabel in dash_widgets.items():
            is_w  = st.session_state.get("active_widget") == wkey
            label = f"● {wlabel}" if is_w else f"  {wlabel}"
            if st.button(label, key=f"nav_{wkey}", width="stretch"):
                st.session_state["active_widget"] = None if is_w else wkey
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    run_btn = st.button(
        "⚡ Generate Forecast",
        width="stretch",
        help="Run the ML model for the selected horizon",
    )

    if fc_is_active:
        st.markdown("""
        <div style="margin:0.3rem 0 0.2rem 0.5rem;padding:0.7rem 0.9rem;
                    background:rgba(96,165,250,0.07);">
        """, unsafe_allow_html=True)

        fc_widgets = {
            "segments_fc": "🏷  Market Segments",
            "yoy_fc":      "📅  Year-over-Year",
            "whatif":      "🎛  What-If Simulator",
            "staff":       "👥  Staff & Inventory",
        }
        for wkey, wlabel in fc_widgets.items():
            is_w  = st.session_state.get("active_widget") == wkey
            label = f"● {wlabel}" if is_w else f"  {wlabel}"
            if st.button(label, key=f"nav_{wkey}", width="stretch"):
                st.session_state["active_widget"] = None if is_w else wkey
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(255,255,255,0.06);margin:0.75rem 0;'>",
                unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# BUTTON CLICK HANDLERS
# ─────────────────────────────────────────────────────────────────────────────
if dash_btn:
    if st.session_state["active_section"] == "dashboard":
        st.session_state["active_section"] = None
        st.session_state["active_widget"]  = None
    else:
        with st.spinner("Loading property data…"):
            try:
                hist = load_full_history(property_choice)
                st.session_state["history_df"]     = hist
                st.session_state["dash_property"]  = property_choice
                st.session_state["active_section"] = "dashboard"
                st.session_state["active_widget"]  = None
                st.session_state.pop("forecast_df", None)
            except Exception as e:
                st.error(f"❌ Could not load history: {e}")
    st.rerun()

if run_btn:
    if st.session_state["active_section"] == "forecast":
        st.session_state["active_section"] = None
        st.session_state["active_widget"]  = None
        st.rerun()
    else:
        with st.spinner("Running forecast engine…"):
            try:
                df_fc = run_csv_forecast(property_choice, int(horizon))
                if df_fc.empty:
                    st.error("❌ Forecast engine returned empty data.")
                    st.stop()
                df_fc["Date"] = pd.to_datetime(df_fc["Date"])
                st.session_state["forecast_df"]    = df_fc
                st.session_state["fc_property"]    = property_choice
                st.session_state["fc_horizon"]     = int(horizon)
                st.session_state["active_section"] = "forecast"
                st.session_state["active_widget"]  = None
                st.session_state.pop("history_df", None)

                # Load & harmonize MS forecast immediately
                _inv  = int(df_fc["Inventory"].iloc[0])
                ms_df = load_ms_forecast(property_choice)
                if not ms_df.empty:
                    harm_df = harmonize_forecasts(df_fc, ms_df, _inv, property_choice)
                    ops_df  = calculate_operations(harm_df)
                    st.session_state["harmonized_df"] = harm_df
                    st.session_state["ops_df"]        = ops_df
                    st.session_state["ms_loaded"]     = True
                else:
                    st.session_state["harmonized_df"] = None
                    st.session_state["ops_df"]        = None
                    st.session_state["ms_loaded"]     = False

            except Exception as e:
                st.error(f"❌ Forecast error: {e}")
                st.stop()
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# EMPTY STATE
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state["active_section"] is None:
    st.markdown("""
    <div style="margin-top:3rem;text-align:center;padding:4rem 2rem;
                background:linear-gradient(135deg,rgba(96,165,250,0.07),rgba(23,195,178,0.07));
                backdrop-filter:blur(12px);border:1px solid rgba(255,255,255,0.08);
                border-radius:20px;">
      <div style="font-size:3.5rem;margin-bottom:1rem;">🏨</div>
      <div style="font-size:1.8rem;font-weight:800;
                  background:linear-gradient(135deg,#17C3B2,#60A5FA);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                  margin-bottom:0.75rem;">Hospify Forecaster</div>
      <p style="color:#8892B0;font-size:1rem;max-width:420px;margin:0 auto 2rem auto;line-height:1.7;">
        Select a property in the sidebar, then click<br>
        <strong style="color:#17C3B2;">Property Dashboard</strong> for historical analysis or<br>
        <strong style="color:#17C3B2;">Generate Forecast</strong> to run the ML model.
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — PROPERTY DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
if (
    st.session_state["active_section"] == "dashboard"
    and "history_df" in st.session_state
    and st.session_state.get("dash_property") == property_choice
):
    hist      = st.session_state["history_df"]
    prop      = property_choice
    inventory = int(PROPERTY_CFG[base_key]["inventory"])

    current_year = 2026
    cy_hist = hist[hist["Dates"].dt.year == current_year].copy()
    py_hist = hist[hist["Dates"].dt.year == current_year - 1].copy()
    has_adr = "ADR" in hist.columns

    cy_occ   = _safe_mean(cy_hist["Occperc"]) if not cy_hist.empty else 0.0
    py_occ   = _safe_mean(py_hist["Occperc"]) if not py_hist.empty else 0.0
    cy_rooms = int((cy_hist["Occperc"] / 100 * cy_hist["Inventory"]).sum()) if not cy_hist.empty else 0
    py_rooms = int((py_hist["Occperc"] / 100 * py_hist["Inventory"]).sum()) if not py_hist.empty else 0

    if has_adr and not cy_hist.empty:
        cy_adr    = _safe_mean(cy_hist["ADR"])
        py_adr    = _safe_mean(py_hist["ADR"]) if not py_hist.empty else 0.0
        cy_revpar = round(cy_occ / 100 * cy_adr, 2)
        py_revpar = round(py_occ / 100 * py_adr, 2)
        cy_rev    = float((cy_hist["ADR"] * cy_hist["Occperc"] / 100 * cy_hist["Inventory"]).sum())
        py_rev    = float((py_hist["ADR"] * py_hist["Occperc"] / 100 * py_hist["Inventory"]).sum()) if not py_hist.empty else 0.0
    else:
        cy_adr = py_adr = cy_revpar = py_revpar = cy_rev = py_rev = 0.0

    last_date_str = cy_hist["Dates"].max().strftime("%d %b %Y") if not cy_hist.empty else "—"
    st.markdown(f"""
    <div class="page-title">🏨 {prop} — 2026 YTD Dashboard</div>
    <div class="page-sub">
      Historical data through {last_date_str} ·
      {inventory} rooms · Comparing {current_year} vs {current_year - 1}
    </div>
    """, unsafe_allow_html=True)

    # ── KPI Grid ─────────────────────────────────────────────────────────────
    kpi_html  = '<div class="kpi-grid">'
    kpi_html += kpi_card(
        "Avg Occupancy 2026", f"{cy_occ:.1f}%",
        f"vs {py_occ:.1f}% in 2025  {_delta_color(cy_occ - py_occ)} {cy_occ - py_occ:+.1f}pp",
        "kpi-teal", "📊",
    )
    kpi_html += kpi_card(
        "Total Rooms Sold", f"{cy_rooms:,}",
        f"vs {py_rooms:,} in 2025  {_delta_color(cy_rooms - py_rooms)} {cy_rooms - py_rooms:+,}",
        "kpi-amber", "🏠",
    )
    if has_adr:
        kpi_html += kpi_card(
            "Avg ADR 2026", f"${cy_adr:,.2f}",
            f"vs ${py_adr:,.2f} in 2025  {_delta_color(cy_adr - py_adr)} ${cy_adr - py_adr:+.2f}",
            "kpi-coral", "💳",
        )
        kpi_html += kpi_card(
            "RevPAR 2026", f"${cy_revpar:,.2f}",
            f"vs ${py_revpar:,.2f} in 2025  {_delta_color(cy_revpar - py_revpar)} ${cy_revpar - py_revpar:+.2f}",
            "kpi-green", "💰",
        )
        kpi_html += kpi_card(
            "Total Revenue 2026", f"${cy_rev:,.0f}",
            f"vs ${py_rev:,.0f} in 2025  {_delta_color(cy_rev - py_rev)} ${cy_rev - py_rev:+,.0f}",
            "kpi-teal", "📈",
        )
    else:
        kpi_html += kpi_card("Inventory", f"{inventory}", "ADR coming soon", "kpi-coral", "🔑")
    kpi_html += "</div>"
    st.markdown(kpi_html, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if st.session_state["active_widget"] is None:
        st.markdown("""
        <div style="padding:1.5rem;text-align:center;
                    background:rgba(23,195,178,0.05);
                    border:1px dashed rgba(23,195,178,0.25);
                    border-radius:14px;color:#64748B;font-size:0.9rem;">
          👈 &nbsp; Select a widget from the sidebar to explore historical data
        </div>
        """, unsafe_allow_html=True)

    # ── WIDGET: YOY Trend ─────────────────────────────────────────────────────
    if st.session_state["active_widget"] == "yoy_trend":
        st.markdown("""
        <div style="font-size:1rem;font-weight:700;color:#E2E8F0;margin-bottom:0.75rem;
                    padding-bottom:0.5rem;border-bottom:1px solid rgba(23,195,178,0.2);">
          📅 YOY Trend Chart
        </div>""", unsafe_allow_html=True)
        section_header("Year-over-Year Revenue · Rooms · ADR")

        col_m, col_metric, _ = st.columns([1, 2, 3])
        with col_m:
            month_names      = {i: calendar.month_name[i] for i in range(1, 13)}
            available_months = sorted(
                hist[hist["Dates"].dt.year.isin([current_year, current_year - 1])]["Dates"]
                .dt.month.unique()
            )
            sel_month = st.selectbox(
                "Month", options=available_months,
                format_func=lambda m: month_names[m],
                key="yoy_month",
            )
        with col_metric:
            sel_metric = st.radio(
                "Metric", options=["Revenue", "Rooms Sold", "ADR"],
                horizontal=True, key="yoy_metric",
            )

        if not has_adr and sel_metric in ["Revenue", "ADR"]:
            st.info("ℹ️ ADR column not found in history CSV. Add 'ADR' column to enable Revenue and ADR charts.")
        else:
            fig_yoy = chart_yoy_revenue(hist, sel_month, sel_metric, current_year)
            st.plotly_chart(fig_yoy, width="stretch")

        section_header("YOY Trend Table")
        yoy_df_table = build_yoy_table(hist, sel_month, current_year)
        # Render as glass HTML table (transparent, consistent with other tables)
        st.markdown(render_glass_table(yoy_df_table), unsafe_allow_html=True)

    # ── WIDGET: Market Segments (Dashboard — historical bar + donut) ──────────
    if st.session_state["active_widget"] == "segments":
        st.markdown("""
        <div style="font-size:1rem;font-weight:700;color:#E2E8F0;margin-bottom:0.75rem;
                    padding-bottom:0.5rem;border-bottom:1px solid rgba(23,195,178,0.2);">
          🏷 Market Segments
        </div>""", unsafe_allow_html=True)
        section_header("Historical Segment Contribution")
        st.caption(
            "Based on historical booking records. Shows which segments have driven "
            "the most room-nights for this property over time."
        )

        ms_hist_path = _MS_HISTORY_CSV.get(prop, "")
        fig_bar = chart_segment_history_bar(ms_hist_path, prop)
        if fig_bar.data:
            st.plotly_chart(fig_bar, width="stretch")
        else:
            st.info("ℹ️ Historical segment CSV not found. Place it in the data/ folder.")

        section_header("Segment Distribution — Donut")

        # Build a lightweight donut from the same CSV
        if ms_hist_path and os.path.exists(ms_hist_path):
            try:
                ms_hist_df = pd.read_csv(ms_hist_path)
                by_seg     = ms_hist_df.groupby("MarketSegment")["RoomNight"].sum().reset_index()
                by_seg.columns = ["Date", "Rooms"]   # reuse donut helper via dummy df

                # The donut helper reads _get_segment_cols → build a pivot-style df
                pivot_df = ms_hist_df.groupby("MarketSegment")["RoomNight"].sum()
                dummy_df = pd.DataFrame([pivot_df.to_dict()])
                dummy_df["Date"] = "Historical"

                st.plotly_chart(chart_segment_donut(dummy_df), width="stretch")

                # Insight text
                top_seg  = pivot_df.idxmax()
                top_pct  = pivot_df.max() / pivot_df.sum() * 100
                sec_seg  = pivot_df.drop(top_seg).idxmax()
                sec_pct  = pivot_df.drop(top_seg).max() / pivot_df.sum() * 100
                st.markdown(f"""
                <div class="insight-card" style="margin-top:1rem;">
                  <div class="insight-icon">💡</div>
                  <div>
                    <div class="insight-title">Segment Insight — {prop}</div>
                    <div class="insight-text">
                      <b>{top_seg}</b> is the dominant channel at <b>{top_pct:.1f}%</b> of all
                      historical room-nights, followed by <b>{sec_seg}</b> at <b>{sec_pct:.1f}%</b>.
                      Combined, these two segments represent
                      <b>{top_pct + sec_pct:.1f}%</b> of total demand.
                      Revenue strategy should prioritise protecting the rate integrity of
                      the leading segment while developing the secondary channel.
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"Could not render segment donut: {e}")

    # ── WIDGET: Validation ────────────────────────────────────────────────────
    if st.session_state["active_widget"] == "validation":
        st.markdown("""
        <div style="font-size:1rem;font-weight:700;color:#E2E8F0;margin-bottom:0.75rem;
                    padding-bottom:0.5rem;border-bottom:1px solid rgba(23,195,178,0.2);">
          ✅ Model Validation
        </div>""", unsafe_allow_html=True)
        try:
            val_df = load_val_csv(prop)
            if val_df.empty:
                st.warning("No validation data found.")
            else:
                fig_v, mae_pct, mae_rooms = chart_validation(val_df)
                c1, c2, c3, c4 = st.columns(4)
                with c1: st.metric("MAE (Occ %)",  f"{mae_pct:.2f}%")
                with c2: st.metric("MAE (Rooms)",  f"{mae_rooms:.1f}")
                with c3:
                    within_7 = (val_df["Absolute_Error_Rooms"] <= 7).mean() * 100
                    st.metric("Within 7-Room Target", f"{within_7:.0f}%")
                with c4:
                    smape = (
                        2 * val_df["Absolute_Error_%"]
                        / (val_df["Actual_Occperc"].abs() + val_df["Predicted_Occperc"].abs() + 1e-9)
                    ).mean() * 100
                    st.metric("SMAPE", f"{smape:.1f}%")
                st.plotly_chart(fig_v, width="stretch")
        except Exception as e:
            st.error(f"Could not load validation: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — FORECAST
# ─────────────────────────────────────────────────────────────────────────────
if (
    st.session_state["active_section"] == "forecast"
    and "forecast_df" in st.session_state
    and st.session_state.get("fc_property") == property_choice
):
    results_df = st.session_state["forecast_df"]
    harmonized = st.session_state.get("harmonized_df")
    ops_df     = st.session_state.get("ops_df")
    ms_loaded  = st.session_state.get("ms_loaded", False)
    prop       = st.session_state["fc_property"]
    hz         = st.session_state["fc_horizon"]
    inventory  = int(results_df["Inventory"].iloc[0])
    prop_cfg   = PROPERTY_CFG.get(base_key, {})
    segments   = prop_cfg.get("ms_segments", [])

    has_adr_fc  = "ADR" in results_df.columns and results_df["ADR"].sum() > 0
    avg_occ     = round(results_df["Predicted_Occperc"].mean(), 1)
    peak_row    = results_df.loc[results_df["Predicted_Occperc"].idxmax()]
    low_row     = results_df.loc[results_df["Predicted_Occperc"].idxmin()]
    peak_date   = peak_row["Date"].strftime("%b %d")
    peak_occ    = round(float(peak_row["Predicted_Occperc"]), 1)
    low_occ     = round(float(low_row["Predicted_Occperc"]), 1)
    total_rns   = int(
        results_df["RoomsSold"].sum()
        if "RoomsSold" in results_df.columns
        else results_df["Predicted_Rooms"].sum()
    )
    start_month = results_df["Date"].dt.month.iloc[0]
    fc_start    = results_df["Date"].iloc[0].strftime("%d %b %Y")
    fc_end      = results_df["Date"].iloc[-1].strftime("%d %b %Y")

    avg_adr    = round(results_df["ADR"].mean(), 2)    if has_adr_fc else 0.0
    avg_revpar = round(results_df["RevPAR"].mean(), 2) if "RevPAR" in results_df.columns else 0.0
    total_rev  = round(results_df["Revenue"].sum(), 0) if "Revenue" in results_df.columns else 0.0

    wk_avg      = results_df[results_df["Date"].dt.dayofweek < 5]["Predicted_Occperc"].mean()
    we_avg      = results_df[results_df["Date"].dt.dayofweek >= 5]["Predicted_Occperc"].mean()
    days_above_80 = int((results_df["Predicted_Occperc"] >= 80).sum())

    st.markdown(
        f'<div class="page-title">⚡ Forecast — {prop}</div>'
        f'<div class="page-sub">{hz}-day outlook · '
        f'{fc_start} → {fc_end} · {inventory} rooms · '
        f'{"📊 MS segments loaded" if ms_loaded else "⚠️ MS segments unavailable"}</div>',
        unsafe_allow_html=True,
    )

    # ── Forecast KPIs (Linen KPI removed as requested) ───────────────────────
    kpi_html  = '<div class="kpi-grid">'
    kpi_html += kpi_card(
        "Avg Occupancy", f"{avg_occ}%",
        f"Peak {peak_occ}% on {peak_date}", "kpi-teal", "📊",
    )
    kpi_html += kpi_card(
        "Total Room-Nights", f"{total_rns:,}",
        f"Avg {round(total_rns / hz, 1)} rooms/night", "kpi-amber", "🏠",
    )
    if has_adr_fc:
        kpi_html += kpi_card(
            "Avg ADR (Forecast)", f"${avg_adr:,.2f}",
            "From same dates last year", "kpi-coral", "💳",
        )
        kpi_html += kpi_card(
            "Avg RevPAR", f"${avg_revpar:,.2f}",
            f"Total est. revenue ${total_rev:,.0f}", "kpi-green", "💰",
        )
    else:
        kpi_html += kpi_card(
            "Inventory", f"{inventory}",
            f"Low {low_occ}% on {low_row['Date'].strftime('%b %d')}",
            "kpi-green", "🔑",
        )

    # Housekeepers KPI only (Linen KPI removed)
    if ms_loaded and ops_df is not None:
        avg_hk = int(ops_df["Required_Housekeepers"].mean())
        avg_fd = int(ops_df["Front_Desk_Agents"].mean())
        kpi_html += kpi_card(
            "Avg Housekeepers/Day", f"{avg_hk}",
            f"Avg front desk: {avg_fd}", "kpi-teal", "🧹",
        )

    kpi_html += "</div>"
    st.markdown(kpi_html, unsafe_allow_html=True)

    # ── Main view (no sub-widget) ─────────────────────────────────────────────
    if st.session_state["active_widget"] is None:
        insight = season_insight(start_month, avg_occ, peak_date, peak_occ, inventory, prop)
        lines   = insight.split("\n\n")
        st.markdown(f"""
        <div class="insight-card">
          <div class="insight-icon">💡</div>
          <div>
            <div class="insight-title">Seasonal Intelligence</div>
            <div class="insight-text">
              {" &nbsp;·&nbsp; ".join(l.replace("**","").replace("*","") for l in lines)}
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        section_header("Occupancy Forecast")
        st.plotly_chart(chart_forecast(results_df, hz), width="stretch")

        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Weekday Average",  f"{wk_avg:.1f}%")
        with col2: st.metric("Weekend Average",  f"{we_avg:.1f}%",  delta=f"{we_avg - wk_avg:+.1f}%")
        with col3: st.metric("Days ≥ 80% Occ",  f"{days_above_80} days",
                             delta=f"{days_above_80 / hz * 100:.0f}% of period")

        st.markdown("<br>", unsafe_allow_html=True)

        # Forecast Data Table (glass)
        section_header("Forecast Data Table")
        base_cols  = ["Date", "DayOfWeek", "Predicted_Occperc", "RoomsSold", "Inventory"]
        extra_cols = [c for c in ["ADR", "Revenue", "RevPAR"] if c in results_df.columns]
        disp = results_df[base_cols + extra_cols].copy()
        disp["Date"] = disp["Date"].dt.strftime("%Y-%m-%d")

        headers = ["Date", "Day", "Occ %", "Rooms", "Inventory"]
        if "ADR"     in extra_cols: headers.append("ADR")
        if "Revenue" in extra_cols: headers.append("Revenue")
        if "RevPAR"  in extra_cols: headers.append("RevPAR")

        html = '<div class="glass-table-container"><table class="glass-table"><thead><tr>'
        html += "".join(f"<th>{h}</th>" for h in headers)
        html += "</tr></thead><tbody>"

        for _, row in disp.iterrows():
            is_we    = row["DayOfWeek"] in ("Saturday", "Sunday")
            row_cls  = "weekend-row" if is_we else ""
            occ_val  = float(row["Predicted_Occperc"])
            opacity  = min(occ_val / 100.0, 1.0) * 0.4
            occ_style = (
                f"background-color:rgba(23,195,178,{opacity:.2f});"
                "border-radius:4px;font-weight:bold;"
            )
            cells = [
                row["Date"], row["DayOfWeek"],
                f'<span style="{occ_style}">{occ_val:.1f}</span>',
                f"{int(row['RoomsSold'])}",
                f"{int(row['Inventory'])}",
            ]
            if "ADR"     in extra_cols: cells.append(f"${row['ADR']:.0f}")
            if "Revenue" in extra_cols: cells.append(f"${row['Revenue']:,.0f}")
            if "RevPAR"  in extra_cols: cells.append(f"${row['RevPAR']:.2f}")
            html += f'<tr class="{row_cls}">' + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"

        html += "</tbody></table></div>"
        st.markdown(html, unsafe_allow_html=True)

        col_dl, col_info = st.columns([1, 3])
        with col_dl:
            csv_bytes = disp.to_csv(index=False).encode("utf-8")
            fname     = f"hospify_{base_key.lower().replace(' ','_')}_{hz}d.csv"
            st.download_button("📥 Download CSV", data=csv_bytes, file_name=fname, mime="text/csv")
        with col_info:
            st.caption(f"Forecast for **{prop}** · {hz} days · {len(results_df)} rows · weekends highlighted.")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="padding:1.25rem;text-align:center;
                    background:rgba(96,165,250,0.05);
                    border:1px dashed rgba(96,165,250,0.25);
                    border-radius:14px;color:#64748B;font-size:0.9rem;">
          👈 &nbsp; Explore deeper analysis using the widgets in the sidebar
        </div>
        """, unsafe_allow_html=True)

    # ── WIDGET A: Market Segments (Forecast — donut + table only, NO heatmap) ─
    if st.session_state["active_widget"] == "segments_fc":
        st.markdown("""
        <div style="font-size:1rem;font-weight:700;color:#E2E8F0;margin-bottom:0.75rem;
                    padding-bottom:0.5rem;border-bottom:1px solid rgba(96,165,250,0.2);">
          🏷 Market Segments
        </div>""", unsafe_allow_html=True)
        section_header("Market Segment Forecast")

        if ms_loaded and harmonized is not None:
            st.caption(
                f"✅ Using real EWMA pickup model · {len(segments)} segments · "
                "rooms harmonized to LightGBM occupancy forecast."
            )

            present_segs = [s for s in segments if s in harmonized.columns]

            m1, m2, m3 = st.columns(3)
            total_rooms_period = int(harmonized["Target_Total_Rooms"].sum())
            avg_occ_h          = round((harmonized["Target_Total_Rooms"] / inventory * 100).mean(), 1)
            dominant_seg       = harmonized[present_segs].sum().idxmax() if present_segs else "—"
            m1.metric("Total Forecasted Rooms",    f"{total_rooms_period:,}", f"Over {hz} days")
            m2.metric("Avg Occupancy (Harmonized)", f"{avg_occ_h}%")
            m3.metric("Dominant Segment",           dominant_seg)

            # Large donut only (heatmap removed from this view)
            st.plotly_chart(chart_segment_donut(harmonized), width="stretch")

            # Segment detail table (glass) — Total Rooms = Σ segments (verified)
            section_header("Segment Detail Table")
            seg_table = harmonized[["Date", "DayOfWeek"] + present_segs].copy()
            # Recompute total as sum of segments — single source of truth
            seg_table["Total Rooms"] = seg_table[present_segs].sum(axis=1).clip(upper=inventory).astype(int)
            seg_table["Date"] = harmonized["Date"].dt.strftime("%Y-%m-%d")

            _hdrs = ["Date", "Day"] + present_segs + ["Total Rooms"]
            _html = '<div class="glass-table-container"><table class="glass-table"><thead><tr>'
            _html += "".join(f"<th>{h}</th>" for h in _hdrs)
            _html += "</tr></thead><tbody>"
            for _, _row in seg_table.iterrows():
                is_we = _row["DayOfWeek"] in ("Saturday", "Sunday")
                _rc   = "weekend-row" if is_we else ""
                _cells = [_row["Date"], _row["DayOfWeek"]]
                _cells += [str(int(_row[s])) for s in present_segs]
                _cells += [f"<b>{int(_row['Total Rooms'])}</b>"]
                _html += f'<tr class="{_rc}">' + "".join(f"<td>{c}</td>" for c in _cells) + "</tr>"
            _html += "</tbody></table></div>"
            st.markdown(_html, unsafe_allow_html=True)

            col_dl2, _ = st.columns([1, 3])
            with col_dl2:
                csv2 = seg_table.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Segment CSV", data=csv2,
                    file_name=f"segments_{base_key.lower().replace(' ','_')}.csv",
                    mime="text/csv",
                )

        else:
            st.caption("⚠️ MS forecast CSV not found — showing static segment mix ratios.")
            seg_df = build_segment_forecast(results_df, prop)
            st.plotly_chart(chart_segment_donut(seg_df), width="stretch")

    # ── WIDGET B: Year-over-Year ──────────────────────────────────────────────
    if st.session_state["active_widget"] == "yoy_fc":
        st.markdown("""
        <div style="font-size:1rem;font-weight:700;color:#E2E8F0;margin-bottom:0.75rem;
                    padding-bottom:0.5rem;border-bottom:1px solid rgba(96,165,250,0.2);">
          📅 Year-over-Year Comparison
        </div>""", unsafe_allow_html=True)
        section_header("Historical Year-over-Year")
        try:
            st.plotly_chart(chart_yoy(prop, results_df), width="stretch")
            fc_month = results_df["Date"].dt.month.iloc[0]
            yoy_df   = fetch_yoy_data(prop, fc_month)
            if "ADR" in yoy_df.columns:
                yoy_df["RevPAR"] = (yoy_df["Occperc"] / 100) * yoy_df["ADR"]
            else:
                yoy_df["ADR"]    = 0
                yoy_df["RevPAR"] = 0

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric(
                    "Forecast Occ %", f"{avg_occ:.1f}%",
                    delta=f"{avg_occ - yoy_df['Occperc'].mean():+.1f}% vs STLY",
                )
            with c2:
                st.metric(
                    "Forecast ADR",
                    f"${avg_adr:.2f}" if has_adr_fc else "—",
                    delta=f"${avg_adr - yoy_df['ADR'].mean():+.2f} vs STLY" if has_adr_fc else None,
                )
            with c3:
                st.metric(
                    "Forecast RevPAR",
                    f"${avg_revpar:.2f}" if has_adr_fc else "—",
                    delta=f"${avg_revpar - yoy_df['RevPAR'].mean():+.2f} vs STLY" if has_adr_fc else None,
                )
        except Exception as e:
            st.error(f"YoY chart error: {e}")

    # ── WIDGET C: Rate What-If Simulator (PED-based) ─────────────────────────
    if st.session_state["active_widget"] == "whatif":
        st.markdown("""
        <div style="font-size:1rem;font-weight:700;color:#E2E8F0;margin-bottom:0.75rem;
                    padding-bottom:0.5rem;border-bottom:1px solid rgba(96,165,250,0.2);">
          🎛 Rate What-If Simulator
        </div>""", unsafe_allow_html=True)
        section_header("Proposed Rate Change → Occupancy Impact")

        st.caption(
            "Adjust the proposed rate change below. The simulator uses property-specific "
            "Price Elasticity of Demand (PED) calibrated per day-of-week from historical data."
        )

        # ── Controls row ─────────────────────────────────────────────────────
        ctrl1, ctrl2, ctrl3 = st.columns([3, 2, 2])
        with ctrl1:
            rate_change = st.slider(
                "💳 Proposed Rate Change (%)",
                min_value=-30, max_value=30, value=0, step=1,
                key="wi_rate",
                help="Positive = rate increase (may reduce occ). Negative = discount (may boost occ).",
            )
        with ctrl2:
            rooms_out = st.slider(
                "🔧 Rooms Under Renovation",
                min_value=0, max_value=inventory // 4, value=0, step=1,
                key="wi_reno_rate",
            )
        with ctrl3:
            # Show base DOW ADR reference table
            from engine import _BASE_ADR, _BASE_OCC, _PED_BASE
            ped_ref = _PED_BASE.get(base_key, {})
            adr_ref = _BASE_ADR.get(base_key, {})
            st.markdown(
                "<div style='font-size:0.72rem;color:#64748B;text-transform:uppercase;"
                "letter-spacing:0.07em;margin-bottom:0.3rem;'>PED Reference (per DOW)</div>",
                unsafe_allow_html=True,
            )
            ped_rows = "".join(
                f"<tr><td style='color:#94A3B8'>{d}</td>"
                f"<td style='color:#60A5FA'>{v:+.2f}</td>"
                f"<td style='color:#CBD5E1'>${adr_ref.get(d, 0):.0f}</td></tr>"
                for d, v in ped_ref.items()
            )
            st.markdown(
                f"<table style='font-size:0.75rem;width:100%;border-collapse:collapse;'>"
                f"<thead><tr>"
                f"<th style='color:#475569;text-align:left'>Day</th>"
                f"<th style='color:#475569;text-align:left'>PED</th>"
                f"<th style='color:#475569;text-align:left'>Base ADR</th>"
                f"</tr></thead><tbody>{ped_rows}</tbody></table>",
                unsafe_allow_html=True,
            )

        # ── Run PED simulation ────────────────────────────────────────────────
        adj_rate_df = apply_rate_whatif(results_df, property_choice, rate_change, rooms_out)

        eff_inventory  = max(1, inventory - rooms_out)
        base_avg_occ   = adj_rate_df["Base_Occperc"].mean()
        adj_avg_occ    = adj_rate_df["Adjusted_Occperc"].mean()
        delta_occ      = adj_avg_occ - base_avg_occ
        base_rev_total = adj_rate_df["Rev_Base"].sum()
        adj_rev_total  = adj_rate_df["Rev_Adjusted"].sum()
        rev_delta      = adj_rev_total - base_rev_total
        base_rns       = (adj_rate_df["Base_Occperc"] / 100 * eff_inventory).round(0).sum()
        adj_rns        = adj_rate_df["Adjusted_Rooms"].sum()
        rn_delta       = int(adj_rns - base_rns)

        # ── KPI row ──────────────────────────────────────────────────────────
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Base Avg Occ",       f"{base_avg_occ:.1f}%")
        k2.metric("Adjusted Avg Occ",   f"{adj_avg_occ:.1f}%",   delta=f"{delta_occ:+.1f}pp")
        k3.metric("Effective Inventory", f"{eff_inventory}",
                  delta=f"{eff_inventory - inventory:+d}" if rooms_out > 0 else None)
        k4.metric("Room-Night Δ",       f"{rn_delta:+,}")
        k5.metric("Revenue Δ",          f"${rev_delta:+,.0f}")

        # ── Chart ─────────────────────────────────────────────────────────────
        if rate_change != 0:
            st.plotly_chart(chart_rate_whatif(adj_rate_df, rate_change), width="stretch")
        else:
            st.info("ℹ️ Move the rate slider above to model a rate change scenario.")

        # ── Day-of-week impact table ──────────────────────────────────────────
        section_header("Day-of-Week Impact Summary")
        dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow_summary = (
            adj_rate_df.groupby("DayOfWeek")
            .agg(
                Base_ADR=("Base_ADR", "mean"),
                Proposed_ADR=("Proposed_ADR", "mean"),
                PED=("PED", "mean"),
                Base_Occ=("Base_Occperc", "mean"),
                Adj_Occ=("Adjusted_Occperc", "mean"),
                Occ_Delta=("Occ_Delta_pp", "mean"),
                Rev_Delta=("Rev_Delta", "sum"),
            )
            .reindex([d for d in dow_order if d in adj_rate_df["DayOfWeek"].unique()])
            .reset_index()
        )

        _hdrs = ["Day", "Base ADR", "Proposed ADR", "PED", "Base Occ%", "Adj Occ%", "Δ Occ pp", "Rev Δ ($)"]
        _html = '<div class="glass-table-container"><table class="glass-table"><thead><tr>'
        _html += "".join(f"<th>{h}</th>" for h in _hdrs)
        _html += "</tr></thead><tbody>"
        for _, r in dow_summary.iterrows():
            delta_cls  = "color:#E63946" if r["Occ_Delta"] < 0 else "color:#2A9D8F"
            rdelta_cls = "color:#E63946" if r["Rev_Delta"] < 0 else "color:#2A9D8F"
            _html += (
                f"<tr>"
                f"<td>{r['DayOfWeek'][:3]}</td>"
                f"<td>${r['Base_ADR']:.0f}</td>"
                f"<td>${r['Proposed_ADR']:.0f}</td>"
                f"<td>{r['PED']:+.2f}</td>"
                f"<td>{r['Base_Occ']:.1f}%</td>"
                f"<td>{r['Adj_Occ']:.1f}%</td>"
                f"<td style='{delta_cls}'>{r['Occ_Delta']:+.2f}pp</td>"
                f"<td style='{rdelta_cls}'>${r['Rev_Delta']:+,.0f}</td>"
                f"</tr>"
            )
        _html += "</tbody></table></div>"
        st.markdown(_html, unsafe_allow_html=True)

        # ── Narrative insight ─────────────────────────────────────────────────
        if rate_change > 0 and rev_delta > 0:
            st.success(
                f"✅ Despite a **{delta_occ:.1f}pp** drop in occupancy, raising rates by "
                f"**{rate_change}%** generates **${rev_delta:+,.0f}** additional revenue. "
                f"Net room-nights: {rn_delta:+,}."
            )
        elif rate_change > 0 and rev_delta <= 0:
            st.warning(
                f"⚠️ Raising rates by **{rate_change}%** causes occupancy to drop by "
                f"**{abs(delta_occ):.1f}pp**, resulting in a net revenue loss of "
                f"**${abs(rev_delta):,.0f}**. Consider a smaller rate increase."
            )
        elif rate_change < 0 and rev_delta >= 0:
            st.success(
                f"✅ The **{rate_change}%** discount stimulates **{delta_occ:+.1f}pp** more "
                f"occupancy and adds **${rev_delta:+,.0f}** in revenue. "
                f"Net room-nights: {rn_delta:+,}."
            )
        elif rate_change < 0 and rev_delta < 0:
            st.warning(
                f"⚠️ The **{rate_change}%** discount adds **{delta_occ:+.1f}pp** occupancy "
                f"but still results in a **${abs(rev_delta):,.0f}** revenue shortfall. "
                f"Demand is not elastic enough to compensate."
            )
        else:
            st.info("No rate change applied. Adjust the slider to model a scenario.")

    # ── WIDGET D: Staff & Inventory ───────────────────────────────────────────
    if st.session_state["active_widget"] == "staff":
        st.markdown("""
        <div style="font-size:1rem;font-weight:700;color:#E2E8F0;margin-bottom:0.75rem;
                    padding-bottom:0.5rem;border-bottom:1px solid rgba(96,165,250,0.2);">
          👥 Staff & Inventory
        </div>""", unsafe_allow_html=True)

        if ms_loaded and ops_df is not None:
            section_header("Operational Requirements (from MS Forecast)")
            st.caption("Staffing requirements calculated from harmonized EWMA segment rooms.")

            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Avg Daily Housekeepers", f"{ops_df['Required_Housekeepers'].mean():.0f}")
            s2.metric("Peak Housekeepers",       f"{ops_df['Required_Housekeepers'].max():.0f}")
            s3.metric("Avg Front Desk Agents",   f"{ops_df['Front_Desk_Agents'].mean():.0f}")
            s4.metric("Peak Front Desk",          f"{ops_df['Front_Desk_Agents'].max():.0f}")

            st.plotly_chart(chart_operations(ops_df), width="stretch")

            section_header("Full Staffing Breakdown (LightGBM Rooms)")
            staff_df = build_staff_plan(results_df)
            s5, s6, s7, s8 = st.columns(4)
            s5.metric("Avg Daily Housekeeping", f"{staff_df['Housekeeping'].mean():.0f}")
            s6.metric("Avg Front Desk",         f"{staff_df['Front Desk'].mean():.0f}")
            s7.metric("Peak Total Staff",       f"{staff_df['Total Staff'].max():.0f}")
            s8.metric("Low-Demand Days",        f"{(staff_df['Occupancy %'] < 40).sum()} days")
            st.plotly_chart(chart_staff(staff_df), width="stretch")

            col_dl3, _ = st.columns([1, 3])
            with col_dl3:
                ops_cols   = ["Date", "DayOfWeek", "Target_Total_Rooms",
                              "Required_Housekeepers", "Front_Desk_Agents", "Linen_Sets_Required"]
                ops_export = ops_df[[c for c in ops_cols if c in ops_df.columns]].copy()
                ops_export["Date"] = ops_export["Date"].dt.strftime("%Y-%m-%d")
                st.download_button(
                    "📥 Download Operations CSV",
                    data=ops_export.to_csv(index=False).encode("utf-8"),
                    file_name=f"operations_{base_key.lower().replace(' ','_')}.csv",
                    mime="text/csv",
                )
        else:
            section_header("Daily Staffing Recommendation")
            st.caption("⚠️ MS forecast CSV not found — using LightGBM room forecast for staffing.")
            staff_df = build_staff_plan(results_df)

            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Avg Daily Housekeeping", f"{staff_df['Housekeeping'].mean():.0f}")
            s2.metric("Avg Front Desk",         f"{staff_df['Front Desk'].mean():.0f}")
            s3.metric("Peak Total Staff",       f"{staff_df['Total Staff'].max():.0f}")
            s4.metric("Low-Demand Days",        f"{(staff_df['Occupancy %'] < 40).sum()} days")
            st.plotly_chart(chart_staff(staff_df), width="stretch")
