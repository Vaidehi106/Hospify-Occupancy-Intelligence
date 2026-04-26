# components.py
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from theme import PLOTLY_LAYOUT, COLORS, SEG_COLORS
from engine import fetch_yoy_data, _SEGMENT_BASE


# ─── Helpers ──────────────────────────────────────────────────────────────────

def season_insight(month: int, avg_occ: float, peak_date: str, peak_occ: float,
                   inventory: int, property_choice: str) -> str:
    if month in [1, 2]:
        season = "Winter / Post-Holiday"
        note   = "Demand typically softens post-holiday. Focus on corporate and extended-stay segments to sustain RevPAR."
    elif month in [3, 4, 5]:
        season = "Spring Shoulder Season"
        note   = "Shoulder demand — strong weekends, softer mid-week. Ideal for targeted rate promotions and group bookings."
    elif month in [6, 7, 8]:
        season = "Peak Summer Season"
        note   = "Sustained leisure demand. Expect high weekend spikes and potential rate sensitivity mid-week."
    elif month in [9, 10, 11]:
        season = "Autumn Conference Season"
        note   = "Business travel recovery expected. MICE and corporate segments should anchor revenue strategy."
    else:
        season = "Holiday Peak"
        note   = "Year-end corporate close-outs and leisure holiday travel combine for volatile, high-occupancy days."

    total_room_nights = round(avg_occ / 100 * inventory * 30)
    return (
        f"**{season}** · Avg forecast {avg_occ}% · Peak {peak_date} at {peak_occ}%\n\n"
        f"{note}\n\n"
        f"Estimated {total_room_nights:,} room-nights over 30 days at {inventory} available rooms."
    )


def kpi_card(label, value, sub="", color_cls="kpi-teal", icon=""):
    return f"""
    <div class="kpi-card {color_cls}">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      {'<div class="kpi-sub">' + sub + '</div>' if sub else ''}
      {'<div class="kpi-icon">' + icon + '</div>' if icon else ''}
    </div>"""


def section_header(label: str) -> None:
    st.markdown(f'<div class="section-header">{label}</div>', unsafe_allow_html=True)


# ─── Glass HTML table helper ───────────────────────────────────────────────────

def render_glass_table(df: pd.DataFrame, weekend_col: str = None) -> str:
    """
    Convert a DataFrame to a glass-styled HTML table string.

    Parameters
    ----------
    df          : DataFrame to render
    weekend_col : column name containing day-of-week strings. Rows where value
                  is 'Saturday' or 'Sunday' get the 'weekend-row' CSS class.
    """
    headers = list(df.columns)
    html = '<div class="glass-table-container"><table class="glass-table"><thead><tr>'
    html += "".join(f"<th>{h}</th>" for h in headers)
    html += "</tr></thead><tbody>"

    for _, row in df.iterrows():
        is_we = False
        if weekend_col and weekend_col in df.columns:
            is_we = str(row[weekend_col]) in ("Saturday", "Sunday")
        row_cls = "weekend-row" if is_we else ""
        html += f'<tr class="{row_cls}">'
        for h in headers:
            html += f"<td>{row[h]}</td>"
        html += "</tr>"

    html += "</tbody></table></div>"
    return html


# ─── Chart builders ────────────────────────────────────────────────────────────

def chart_forecast(df: pd.DataFrame, horizon: int) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Predicted_Occperc"],
        mode="lines", name="Occupancy %",
        fill="tozeroy", fillcolor="rgba(23,195,178,0.08)",
        line=dict(color=COLORS["teal"], width=2.5),
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>Occupancy: %{y:.1f}%<extra></extra>",
    ))
    for _, row in df.iterrows():
        if row["Date"].dayofweek >= 5:
            fig.add_vrect(
                x0=row["Date"] - pd.Timedelta(hours=12),
                x1=row["Date"] + pd.Timedelta(hours=12),
                fillcolor="rgba(10,25,47,0.04)", line_width=0, layer="below",
            )
    fig.add_hline(y=80, line_dash="dot", line_color=COLORS["amber"],
                  annotation_text="80% target", annotation_position="right")
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=f"<b>{horizon}-Day Occupancy Forecast</b>", font=dict(size=15)),
        xaxis_title="Date",
        yaxis=dict(title="Occupancy (%)", range=[0, 105]),
        height=380,
    )
    return fig


def chart_validation(val_df: pd.DataFrame) -> tuple:
    """
    3-panel validation chart:
      Row 1 — Actual vs Predicted occupancy % (line+fill)
      Row 2 — Forecast error (% points) as diverging bars with zero baseline
      Row 3 — Room error: scatter with rolling-7-day average ribbon + 7-room band
    """
    mae_pct   = val_df["Absolute_Error_%"].mean()
    mae_rooms = val_df["Absolute_Error_Rooms"].mean()

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.07,
        row_heights=[0.40, 0.28, 0.32],
        subplot_titles=(
            "Actual vs Predicted Occupancy (%)",
            "Forecast Error (pp) — Over / Under",
            "Room Error Trend vs 7-Room Target",
        ),
    )

    # ── Row 1: Actual vs Predicted — filled area between lines ────────────────
    fig.add_trace(go.Scatter(
        x=val_df["Dates"], y=val_df["Actual_Occperc"],
        mode="lines", name="Actual",
        line=dict(color=COLORS["primary"], width=2.5),
        hovertemplate="<b>Actual</b>: %{y:.1f}%<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=val_df["Dates"], y=val_df["Predicted_Occperc"],
        mode="lines", name="Predicted",
        fill="tonexty",
        fillcolor="rgba(23,195,178,0.12)",
        line=dict(color=COLORS["teal"], width=2, dash="dot"),
        hovertemplate="<b>Predicted</b>: %{y:.1f}%<extra></extra>",
    ), row=1, col=1)

    # ── Row 2: Diverging error bars (over=coral, under=green) ─────────────────
    diff_vals   = val_df["Difference_%"].values
    diff_colors = [
        "rgba(230,57,70,0.80)" if v > 0 else "rgba(42,157,143,0.80)"
        for v in diff_vals
    ]
    fig.add_trace(go.Bar(
        x=val_df["Dates"], y=diff_vals,
        marker_color=diff_colors,
        name="Error (pp)",
        showlegend=False,
        hovertemplate="<b>%{x|%b %d}</b><br>Error: %{y:+.2f}pp<extra></extra>",
    ), row=2, col=1)
    # Zero line
    fig.add_hline(y=0, line_width=1.2, line_color="rgba(148,163,184,0.5)", row=2, col=1)

    # ── Row 3: Room error — scatter + rolling average line + 7-room band ──────
    abs_rooms = val_df["Absolute_Error_Rooms"].values
    rolling7  = pd.Series(abs_rooms).rolling(7, min_periods=1, center=True).mean().values

    # Coloured scatter: green ≤7, amber ≤12, coral >12
    point_colors = [
        COLORS["green"] if v <= 7 else COLORS["amber"] if v <= 12 else COLORS["coral"]
        for v in abs_rooms
    ]
    fig.add_trace(go.Scatter(
        x=val_df["Dates"], y=abs_rooms,
        mode="markers",
        marker=dict(color=point_colors, size=6, opacity=0.85,
                    line=dict(color="rgba(10,25,47,0.3)", width=0.8)),
        name="Daily Room Error",
        showlegend=True,
        hovertemplate="<b>%{x|%b %d}</b><br>Room error: %{y:.1f}<extra></extra>",
    ), row=3, col=1)

    # 7-day rolling average line
    fig.add_trace(go.Scatter(
        x=val_df["Dates"], y=rolling7,
        mode="lines",
        line=dict(color=COLORS["amber"], width=2.2),
        name="7-Day Rolling Avg",
        hovertemplate="<b>Rolling avg</b>: %{y:.1f} rooms<extra></extra>",
    ), row=3, col=1)

    # Shaded tolerance band (0–7 rooms = acceptable zone)
    fig.add_hrect(
        y0=0, y1=7,
        fillcolor="rgba(42,157,143,0.08)",
        line_width=0,
        row=3, col=1,
    )
    fig.add_hline(
        y=7, line_dash="dash", line_color=COLORS["teal"], line_width=1.5,
        annotation_text="7-room threshold",
        annotation_font=dict(color=COLORS["teal"], size=11),
        annotation_position="top right",
        row=3, col=1,
    )

    val_layout = {**PLOTLY_LAYOUT}
    val_layout["legend"] = dict(orientation="h", yanchor="top", y=-0.06, xanchor="center", x=0.5)
    fig.update_layout(
        **val_layout,
        height=760,
        showlegend=True,
    )
    fig.update_annotations(font=dict(color="#94A3B8", size=13))
    fig.update_yaxes(title_text="Occ %", row=1, col=1,
                     gridcolor="rgba(255,255,255,0.04)", range=[0, 105])
    fig.update_yaxes(title_text="Error (pp)", row=2, col=1,
                     gridcolor="rgba(255,255,255,0.04)", zeroline=False)
    fig.update_yaxes(title_text="Rooms", row=3, col=1,
                     gridcolor="rgba(255,255,255,0.04)", rangemode="tozero")
    return fig, round(mae_pct, 2), round(mae_rooms, 1)


def chart_yoy(property_choice: str, forecast_df: pd.DataFrame) -> go.Figure:
    forecast_df  = forecast_df.copy()
    forecast_df["Day"] = forecast_df["Date"].dt.day
    month        = forecast_df["Date"].dt.month.iloc[0]
    year         = forecast_df["Date"].dt.year.iloc[0]
    month_name   = forecast_df["Date"].dt.strftime("%B").iloc[0]

    hist = fetch_yoy_data(property_choice, month)
    hist["Year"] = hist["Dates"].dt.year
    hist["Day"]  = hist["Dates"].dt.day

    fig          = go.Figure()
    grey_palette = [COLORS["grey1"], COLORS["grey2"], COLORS["grey3"], COLORS["grey4"]]
    years        = sorted(hist["Year"].unique())

    for i, y in enumerate(years):
        if y == year:
            continue
        yd = hist[hist["Year"] == y].sort_values("Day")
        fig.add_trace(go.Scatter(
            x=yd["Day"], y=yd["Occperc"],
            mode="lines", name=f"{y}",
            line=dict(width=1.8, dash="dot", color=grey_palette[i % len(grey_palette)]),
            hovertemplate=f"<b>{y}</b> Day %{{x}}: %{{y:.1f}}%<extra></extra>",
        ))

    fc = forecast_df.sort_values("Day")
    fig.add_trace(go.Scatter(
        x=fc["Day"], y=fc["Predicted_Occperc"],
        mode="lines+markers", name=f"Forecast {year}",
        line=dict(color=COLORS["teal"], width=3),
        marker=dict(size=5),
        hovertemplate=f"<b>Forecast {year}</b> Day %{{x}}: %{{y:.1f}}%<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=f"<b>Year-over-Year — {month_name}</b>", font=dict(size=15)),
        xaxis=dict(title=f"Day of {month_name}", dtick=2),
        yaxis=dict(title="Occupancy (%)", range=[0, 105]),
        height=400,
    )
    return fig


def chart_yoy_revenue(hist_df: pd.DataFrame, month: int, metric: str, current_year: int) -> go.Figure:
    df = hist_df.copy()
    df["Dates"]    = pd.to_datetime(df["Dates"])
    df["Year"]     = df["Dates"].dt.year
    df["Month"]    = df["Dates"].dt.month
    df["Day"]      = df["Dates"].dt.day

    if "ADR" not in df.columns:
        df["ADR"] = 0.0
    df["RoomsSold"] = (df["Occperc"] / 100 * df["Inventory"]).round(0)
    df["Revenue"]   = (df["ADR"] * df["RoomsSold"]).round(2)

    month_df = df[df["Month"] == month].copy()
    METRIC_COL = {"Revenue": "Revenue", "Rooms Sold": "RoomsSold", "ADR": "ADR"}
    col = METRIC_COL.get(metric, "Revenue")

    prior_year = current_year - 1
    fig   = go.Figure()
    years = [prior_year, current_year]
    colors = {prior_year: "rgba(96,165,250,0.55)", current_year: "rgba(23,195,178,0.75)"}
    fills  = {prior_year: "rgba(96,165,250,0.15)", current_year: "rgba(23,195,178,0.15)"}

    for yr in years:
        yd = month_df[month_df["Year"] == yr].sort_values("Day").copy()
        if yd.empty:
            continue
        if metric == "ADR":
            yd["plot_val"] = yd[col].expanding().mean()
            y_label = "Running Avg ADR ($)"
        else:
            yd["plot_val"] = yd[col].cumsum()
            y_label = f"Cumulative {metric}"

        fig.add_trace(go.Scatter(
            x=yd["Day"], y=yd["plot_val"],
            mode="lines+markers", name=str(yr),
            fill="tozeroy", fillcolor=fills[yr],
            line=dict(color=colors[yr], width=2.5),
            marker=dict(size=5, color=colors[yr]),
            hovertemplate=f"<b>{yr}</b> Day %{{x}}: %{{y:,.1f}}<extra></extra>",
        ))

    month_name = pd.Timestamp(f"{current_year}-{month:02d}-01").strftime("%B")
    fig.update_layout(
        **{k: v for k, v in PLOTLY_LAYOUT.items() if k != "legend"},
        title=dict(text=f"<b>YOY Trend — {month_name} · {metric}</b>", font=dict(size=14)),
        xaxis=dict(title=f"Day of {month_name}", dtick=2),
        yaxis=dict(title=y_label),
        height=360,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def build_yoy_table(hist_df: pd.DataFrame, month: int, current_year: int) -> pd.DataFrame:
    df = hist_df.copy()
    df["Dates"]  = pd.to_datetime(df["Dates"])
    df["Year"]   = df["Dates"].dt.year
    df["Month"]  = df["Dates"].dt.month
    df["Day"]    = df["Dates"].dt.day

    if "ADR" not in df.columns:
        df["ADR"] = 0.0
    df["RoomsSold"] = (df["Occperc"] / 100 * df["Inventory"]).round(0)
    df["Revenue"]   = (df["ADR"] * df["RoomsSold"]).round(2)

    month_df   = df[df["Month"] == month]
    prior_year = current_year - 1

    cy = month_df[month_df["Year"] == current_year][["Day", "Revenue", "RoomsSold", "ADR"]].rename(
        columns={"Revenue": "Rev_CY", "RoomsSold": "RMS_CY", "ADR": "ADR_CY"})
    py = month_df[month_df["Year"] == prior_year][["Day", "Revenue", "RoomsSold", "ADR"]].rename(
        columns={"Revenue": "Rev_PY", "RoomsSold": "RMS_PY", "ADR": "ADR_PY"})
    merged = pd.merge(cy, py, on="Day", how="outer").sort_values("Day")

    month_name = pd.Timestamp(f"{current_year}-{month:02d}-01").strftime("%b")
    merged["DATE"] = merged["Day"].apply(
        lambda d: f"{month_name} {int(d):02d}/{current_year}" if not pd.isna(d) else "")

    for c in ["Rev_CY", "Rev_PY"]:
        merged[c] = merged[c].apply(lambda x: f"${x:,.0f}" if not pd.isna(x) else "$0")
    for c in ["RMS_CY", "RMS_PY"]:
        merged[c] = merged[c].apply(lambda x: f"{int(x):,}" if not pd.isna(x) else "0")
    for c in ["ADR_CY", "ADR_PY"]:
        merged[c] = merged[c].apply(lambda x: f"${x:.0f}" if not pd.isna(x) else "$0")

    merged = merged[["DATE", "Rev_CY", "Rev_PY", "RMS_CY", "RMS_PY", "ADR_CY", "ADR_PY"]]
    merged.columns = [
        "DATE",
        f"REVENUE {current_year}", f"REVENUE {prior_year}",
        f"ROOMS {current_year}",   f"ROOMS {prior_year}",
        f"ADR {current_year}",     f"ADR {prior_year}",
    ]
    return merged.reset_index(drop=True)


# ─── Segment helpers ───────────────────────────────────────────────────────────

def _get_segment_cols(seg_df: pd.DataFrame) -> list:
    """Dynamically detect segment columns from a DataFrame."""
    legacy_segs = ["Business", "Leisure", "Group", "OTA / Online"]
    if all(s in seg_df.columns for s in legacy_segs):
        return legacy_segs
    skip = {
        "Date", "StayDate", "DayOfWeek", "Predicted_Occperc", "Predicted_Rooms",
        "RoomsSold", "RoomsSold_x", "RoomsSold_y", "Inventory", "Target_Total_Rooms",
        "ADR", "Revenue", "RevPAR", "Required_Housekeepers", "Front_Desk_Agents",
        "Linen_Sets_Required", "Adjusted_Occperc", "Adjusted_Rooms",
        "Base_ADR", "Proposed_ADR", "PED", "Base_Occperc", "Occ_Delta_pp",
        "Rev_Base", "Rev_Adjusted", "Rev_Delta",
    }
    return [c for c in seg_df.columns if c not in skip and pd.api.types.is_numeric_dtype(seg_df[c])]


# ─── NEW: Historical segment horizontal bar chart ──────────────────────────────

def chart_segment_history_bar(ms_csv_path: str, property_name: str) -> go.Figure:
    """
    Horizontal bar chart showing each segment's total historical room-night
    contribution. Used in Property Dashboard → Market Segments widget.

    Parameters
    ----------
    ms_csv_path   : path to MarketSegment_historical_data_<property>.csv
    property_name : display name e.g. 'Aureon'

    Expected CSV columns: StayDate, MarketSegment, RoomNight
    """
    import os
    if not ms_csv_path or not os.path.exists(ms_csv_path):
        return go.Figure()

    try:
        df = pd.read_csv(ms_csv_path)
    except Exception:
        return go.Figure()

    if "MarketSegment" not in df.columns or "RoomNight" not in df.columns:
        return go.Figure()

    by_seg  = df.groupby("MarketSegment")["RoomNight"].sum().sort_values(ascending=True)
    total   = by_seg.sum()
    pcts    = (by_seg / total * 100).round(1)

    palette = [
        "#60A5FA", "#17C3B2", "#F4A261", "#E63946", "#2A9D8F",
        "#A78BFA", "#FB923C", "#34D399", "#F472B6", "#38BDF8",
        "#FBBF24", "#86EFAC", "#C084FC",
    ]
    n       = len(by_seg)
    colors  = (palette * 4)[:n]

    # customdata: [room_nights, pct]
    customdata = list(zip(by_seg.values, pcts.values))

    fig = go.Figure(go.Bar(
        x=by_seg.values,
        y=by_seg.index,
        orientation="h",
        marker=dict(
            color=colors,
            line=dict(color="rgba(10,25,47,0.4)", width=1),
        ),
        customdata=customdata,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "%{customdata[0]:,.0f} room-nights<br>"
            "%{customdata[1]:.1f}% of total<extra></extra>"
        ),
        text=[f"{p:.1f}%" for p in pcts.values],
        textposition="outside",
        textfont=dict(color="#94A3B8", size=11),
    ))

    fig.update_layout(
        **{
            **PLOTLY_LAYOUT,
            "margin": dict(l=10, r=60, t=55, b=30),
        },
        title=dict(
            text=f"<b>Historical Segment Contribution — {property_name}</b>",
            font=dict(size=15),
        ),
        xaxis=dict(
            title="Total Room-Nights",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.05)",
        ),
        yaxis=dict(title="", tickfont=dict(size=12, color="#CBD5E1")),
        height=max(320, 38 * n + 80),
        showlegend=False,
    )
    return fig


# ─── Segment donut (kept, used in both dashboard and forecast) ─────────────────

def chart_segment_donut(seg_df: pd.DataFrame) -> go.Figure:
    """Large donut chart — total room-nights per segment over the period."""
    segs = _get_segment_cols(seg_df)
    if not segs:
        return go.Figure()

    palette = [
        "#60A5FA", "#17C3B2", "#F4A261", "#E63946", "#2A9D8F",
        "#A78BFA", "#FB923C", "#34D399", "#F472B6", "#38BDF8",
        "#FBBF24", "#86EFAC",
    ]
    colors      = (palette * 4)[:len(segs)]
    totals      = [max(float(seg_df[s].sum()), 0) for s in segs]
    total_rooms = sum(totals)

    fig = go.Figure(data=go.Pie(
        labels=segs,
        values=totals,
        hole=0.58,
        marker=dict(colors=colors, line=dict(color="rgba(10,25,47,0.5)", width=2)),
        textinfo="label+percent",
        textfont=dict(size=12, color="#E2E8F0"),
        insidetextorientation="radial",
        hovertemplate="<b>%{label}</b><br>%{value:,.0f} room-nights<br>%{percent}<extra></extra>",
        pull=[0.025] * len(segs),
        direction="clockwise",
        sort=True,
    ))

    fig.add_annotation(
        text=(
            f"<b>{total_rooms:,.0f}</b><br>"
            "<span style='font-size:11px;color:#8892B0'>Total Rooms</span>"
        ),
        x=0.5, y=0.5,
        font=dict(size=22, color="#F1F5F9", family="Space Grotesk"),
        showarrow=False,
        xanchor="center", yanchor="middle",
        align="center",
    )

    fig.update_layout(
        **{
            **PLOTLY_LAYOUT,
            "margin": dict(l=20, r=180, t=60, b=20),
            "legend": dict(
                orientation="v", yanchor="middle", y=0.5,
                xanchor="left", x=1.02,
                font=dict(size=12, color="#CBD5E1"),
                bgcolor="rgba(0,0,0,0)",
            ),
        },
        title=dict(text="<b>Segment Distribution — Total Room-Nights</b>", font=dict(size=15)),
        height=520,
        showlegend=True,
    )
    return fig


# ─── Segment heatmap (kept for forecast widget only) ──────────────────────────

def chart_segment_heatmap(seg_df: pd.DataFrame) -> go.Figure:
    """Heatmap — used only in Forecast → Market Segments (short date windows)."""
    segs = _get_segment_cols(seg_df)
    if not segs:
        return go.Figure()

    z_data   = seg_df[segs].T.values
    x_labels = pd.to_datetime(seg_df["Date"]).dt.strftime("%b %d")

    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=x_labels,
        y=segs,
        colorscale="Teal",
        text=[[f"{v:.0f}" for v in row] for row in z_data],
        texttemplate="%{text}",
        textfont=dict(size=12, color="black"),
        hovertemplate="<b>%{x}</b><br>Segment: %{y}<br>Rooms: %{z:.0f}<extra></extra>",
    ))

    fig.update_layout(
        **{**PLOTLY_LAYOUT, "margin": dict(l=10, r=10, t=50, b=40)},
        title=dict(text="<b>Market Segment Heatmap — Rooms by Day</b>", font=dict(size=15)),
        yaxis=dict(autorange="reversed"),
        xaxis=dict(tickangle=-35),
        height=max(280, 60 * len(segs) + 100),
    )
    return fig


def chart_segment_100_pct(seg_df: pd.DataFrame) -> go.Figure:
    """100% Stacked Bar chart — Business Mix composition."""
    segs = _get_segment_cols(seg_df)
    if not segs:
        return go.Figure()

    palette = ["#60A5FA", "#17C3B2", "#F4A261", "#E63946", "#2A9D8F", "#A78BFA", "#FB923C", "#34D399"]
    colors  = (palette * 4)[:len(segs)]
    totals  = seg_df[segs].sum(axis=1).replace(0, 1)

    fig = go.Figure()
    for seg, color in zip(segs, colors):
        pct = (seg_df[seg] / totals) * 100
        fig.add_trace(go.Bar(
            x=seg_df["Date"], y=pct,
            customdata=seg_df[seg],
            name=seg, marker_color=color,
            hovertemplate=f"<b>{seg}</b><br>%{{y:.1f}}% of Mix<br>(%{{customdata:.0f}} rooms)<extra></extra>",
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        barmode="stack",
        title=dict(text="<b>Market Mix Composition (100% Stacked)</b>", font=dict(size=15)),
        yaxis=dict(title="Mix (%)", range=[0, 100]),
        height=400,
    )
    return fig


# ─── Rate / PED What-If chart ─────────────────────────────────────────────────

def chart_rate_whatif(adj_df: pd.DataFrame, rate_change_pct: float) -> go.Figure:
    """
    3-panel chart for the Rate What-If Simulator.
      Row 1 — Base vs Adjusted Occupancy % over time
      Row 2 — Day-of-week average Occ delta (grouped bar)
      Row 3 — Daily Revenue: Base vs Adjusted (grouped bar)
    """
    direction = "▲ Rate Increase" if rate_change_pct > 0 else "▼ Rate Decrease"
    sign_color = COLORS["coral"] if rate_change_pct > 0 else COLORS["teal"]

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=False,
        vertical_spacing=0.09,
        row_heights=[0.38, 0.28, 0.34],
        subplot_titles=(
            f"Occupancy % — Base vs Adjusted ({direction} {abs(rate_change_pct):.1f}%)",
            "Avg Occ Change by Day-of-Week (pp)",
            "Daily Revenue Impact — Base vs Adjusted",
        ),
    )

    # ── Row 1: Base vs Adjusted Occ % ─────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=adj_df["Date"], y=adj_df["Base_Occperc"],
        mode="lines", name="Base Forecast",
        line=dict(color=COLORS["grey2"], width=2, dash="dot"),
        hovertemplate="Base: %{y:.1f}%<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=adj_df["Date"], y=adj_df["Adjusted_Occperc"],
        mode="lines", name="After Rate Adjustment",
        fill="tonexty",
        fillcolor=f"rgba(23,195,178,0.10)" if rate_change_pct < 0 else "rgba(230,57,70,0.08)",
        line=dict(color=sign_color, width=2.5),
        hovertemplate="Adjusted: %{y:.1f}%<extra></extra>",
    ), row=1, col=1)
    fig.add_hline(y=80, line_dash="dot", line_color=COLORS["amber"],
                  annotation_text="80% target", row=1, col=1)

    # ── Row 2: DOW avg delta (grouped bar) ────────────────────────────────────
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_group = (
        adj_df.groupby("DayOfWeek")["Occ_Delta_pp"]
        .mean()
        .reindex([d for d in dow_order if d in adj_df["DayOfWeek"].unique()])
        .reset_index()
    )
    bar_colors = [
        "rgba(230,57,70,0.80)" if v > 0 else "rgba(23,195,178,0.80)"
        for v in dow_group["Occ_Delta_pp"]
    ]
    fig.add_trace(go.Bar(
        x=dow_group["DayOfWeek"].str[:3],
        y=dow_group["Occ_Delta_pp"],
        marker_color=bar_colors,
        name="Occ Δ (pp)",
        showlegend=False,
        hovertemplate="<b>%{x}</b><br>Δ Occ: %{y:+.2f}pp<extra></extra>",
    ), row=2, col=1)
    fig.add_hline(y=0, line_width=1, line_color="rgba(148,163,184,0.4)", row=2, col=1)

    # ── Row 3: Revenue base vs adjusted (grouped bar) ─────────────────────────
    fig.add_trace(go.Bar(
        x=adj_df["Date"], y=adj_df["Rev_Base"],
        name="Base Revenue",
        marker_color="rgba(148,163,184,0.55)",
        hovertemplate="Base Rev: $%{y:,.0f}<extra></extra>",
    ), row=3, col=1)
    fig.add_trace(go.Bar(
        x=adj_df["Date"], y=adj_df["Rev_Adjusted"],
        name="Adjusted Revenue",
        marker_color=f"rgba(23,195,178,0.80)" if rate_change_pct < 0 else "rgba(96,165,250,0.80)",
        hovertemplate="Adj Rev: $%{y:,.0f}<extra></extra>",
    ), row=3, col=1)

    layout = {**PLOTLY_LAYOUT}
    layout["legend"] = dict(orientation="h", yanchor="top", y=-0.04, xanchor="center", x=0.5)
    fig.update_layout(
        **layout,
        barmode="group",
        height=820,
        showlegend=True,
    )
    fig.update_annotations(font=dict(color="#94A3B8", size=13))
    fig.update_yaxes(title_text="Occ %", row=1, col=1,
                     gridcolor="rgba(255,255,255,0.04)", range=[0, 105])
    fig.update_yaxes(title_text="Δ pp", row=2, col=1,
                     gridcolor="rgba(255,255,255,0.04)", zeroline=False)
    fig.update_yaxes(title_text="Revenue ($)", row=3, col=1,
                     gridcolor="rgba(255,255,255,0.04)")
    return fig

def chart_whatif(base_df: pd.DataFrame, adj_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=base_df["Date"], y=base_df["Predicted_Occperc"],
        mode="lines", name="Base Forecast",
        line=dict(color=COLORS["grey2"], width=2, dash="dot"),
        hovertemplate="Base: %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=adj_df["Date"], y=adj_df["Adjusted_Occperc"],
        mode="lines", name="Adjusted Forecast",
        fill="tonexty", fillcolor="rgba(23,195,178,0.1)",
        line=dict(color=COLORS["teal"], width=2.5),
        hovertemplate="Adjusted: %{y:.1f}%<extra></extra>",
    ))
    fig.add_hline(y=80, line_dash="dot", line_color=COLORS["amber"],
                  annotation_text="80% target")
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="<b>What-If Scenario vs. Base Forecast</b>", font=dict(size=15)),
        yaxis=dict(title="Occupancy (%)", range=[0, 105]),
        height=380,
    )
    return fig


def chart_whatif_horizontal_100pct(base_df: pd.DataFrame, adj_df: pd.DataFrame,
                                   segments: list) -> go.Figure:
    """Horizontal 100% stacked bar — Base vs Adjusted segment mix."""
    palette = [
        "#60A5FA", "#17C3B2", "#F4A261", "#E63946", "#2A9D8F",
        "#A78BFA", "#FB923C", "#34D399", "#F472B6", "#38BDF8",
        "#FBBF24", "#86EFAC",
    ]
    colors = (palette * 4)[:len(segments)]

    base_totals = {s: float(base_df[s].sum()) for s in segments if s in base_df.columns}
    adj_totals  = {s: float(adj_df[s].sum())  for s in segments if s in adj_df.columns}
    base_total  = max(sum(base_totals.values()), 1)
    adj_total   = max(sum(adj_totals.values()),  1)

    fig = go.Figure()
    for seg, color in zip(segments, colors):
        b_val = base_totals.get(seg, 0)
        a_val = adj_totals.get(seg, 0)
        fig.add_trace(go.Bar(
            y=["Base Forecast", "Adjusted Scenario"],
            x=[b_val / base_total * 100, a_val / adj_total * 100],
            name=seg,
            orientation="h",
            marker_color=color,
            customdata=[b_val, a_val],
            hovertemplate=(
                f"<b>{seg}</b><br>%{{x:.1f}}% of mix<br>"
                "%{customdata:,.0f} rooms total<extra></extra>"
            ),
        ))

    fig.update_layout(
        **{**PLOTLY_LAYOUT, "margin": dict(l=10, r=10, t=50, b=60)},
        barmode="stack",
        title=dict(text="<b>Segment Mix — Base vs Adjusted Scenario</b>", font=dict(size=15)),
        xaxis=dict(title="Share of Mix (%)", range=[0, 100]),
        yaxis=dict(title=""),
        height=300,
    )
    return fig


# ─── Operations charts ────────────────────────────────────────────────────────

def chart_operations(ops_df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=ops_df["Date"], y=ops_df["Required_Housekeepers"],
        name="Housekeepers", marker_color=COLORS["teal"],
        hovertemplate="<b>Housekeepers</b><br>%{y}<extra></extra>",
    ), secondary_y=False)
    fig.add_trace(go.Bar(
        x=ops_df["Date"], y=ops_df["Front_Desk_Agents"],
        name="Front Desk Agents", marker_color=COLORS["primary"],
        hovertemplate="<b>Front Desk</b><br>%{y}<extra></extra>",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=ops_df["Date"], y=ops_df["Linen_Sets_Required"],
        name="Linen Sets", mode="lines+markers",
        line=dict(color=COLORS["amber"], width=2), marker=dict(size=5),
        hovertemplate="<b>Linen Sets</b><br>%{y}<extra></extra>",
    ), secondary_y=True)

    fig.update_layout(
        **PLOTLY_LAYOUT, barmode="group",
        title=dict(text="<b>Daily Operational Requirements</b>", font=dict(size=15)),
        height=400,
    )
    fig.update_yaxes(title_text="Staff Count", secondary_y=False)
    fig.update_yaxes(title_text="Linen Sets",  secondary_y=True)
    return fig


def chart_staff(staff_df: pd.DataFrame) -> go.Figure:
    roles  = ["Housekeeping", "Front Desk", "F&B / Restaurant", "Maintenance", "Security"]
    colors = [COLORS["primary"], COLORS["teal"], COLORS["amber"], COLORS["coral"], COLORS["grey3"]]
    fig    = go.Figure()
    for role, col in zip(roles, colors):
        fig.add_trace(go.Bar(
            x=staff_df["Date"], y=staff_df[role],
            name=role, marker_color=col,
            hovertemplate=f"<b>{role}</b><br>%{{y}} staff<extra></extra>",
        ))
    fig.update_layout(
        **PLOTLY_LAYOUT, barmode="stack",
        title=dict(text="<b>Daily Staffing Requirements</b>", font=dict(size=15)),
        yaxis_title="Staff Count", xaxis_title="Date", height=380,
    )
    return fig
