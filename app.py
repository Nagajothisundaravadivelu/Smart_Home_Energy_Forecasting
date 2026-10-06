from __future__ import annotations

import hashlib
from html import escape
from io import BytesIO
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from forecasting import (
    FORECAST_STEPS,
    LOOKBACK_STEPS,
    forecast_next_hour,
    load_household_data,
    train_lstm,
)

st.set_page_config(
    page_title="Home Energy Forecast",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_DATA_PATH = (
    Path.home()
    / "Downloads"
    / "individual+household+electric+power+consumption"
    / "household_power_consumption.txt"
)
ACCENT = "#0F766E"

st.markdown(
    """
    <style>
    .stApp {
        min-height: 100dvh;
        background: linear-gradient(145deg, #f4fbfa 0%, #f7f8fc 52%, #fffaf2 100%);
    }
    [data-testid="stAppViewContainer"] {
        min-height: 100dvh;
    }
    [data-testid="stMain"] {
        min-height: 100dvh;
        overflow-y: auto;
    }
    [data-testid="stMainBlockContainer"] {
        height: auto;
        max-height: none;
        overflow: visible;
        padding-top: 0.35rem;
        padding-bottom: 0.35rem;
        gap: 0.55rem;
    }
    [data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stDecoration"] {
        display: none;
    }
    [data-testid="stSidebar"] {
        height: 100dvh;
        overflow-y: auto;
        background: linear-gradient(180deg, #e7f5f1 0%, #f2f6fb 100%);
        border-right: 1px solid #d9e8e5;
        color: #203f3b !important;
    }
    [data-testid="stSidebar"] :is(h1, h2, h3, p, label, small, span) {
        color: #294845 !important;
    }
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        font-size: 0.9rem;
        font-weight: 650;
    }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
        color: #526b6b !important;
        font-size: 0.82rem;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label,
    [data-testid="stSidebar"] [data-testid="stRadio"] label * {
        color: #365955 !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked),
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) * {
        color: #0f625a !important;
    }
    [data-testid="stSidebar"][aria-expanded="false"] {
        width: 300px !important;
        min-width: 300px !important;
        transform: translateX(0) !important;
        visibility: visible !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        height: 100%;
        overflow-y: auto;
    }
    [data-testid="stMetric"] {
        background: transparent;
        border: 0;
        padding: 0;
        box-shadow: none;
    }
    .hero {
        padding: 0.75rem 1.25rem;
        margin: 0.15rem 0 0.5rem 0;
        color: white;
        border-radius: 16px;
        background: linear-gradient(115deg, #0f766e 0%, #168b83 55%, #4f9da0 100%);
        box-shadow: 0 10px 28px rgba(15, 118, 110, 0.18);
    }
    .hero h1 {
        margin: 0 0 0.2rem 0;
        font-size: 1.55rem;
    }
    .hero p {
        margin: 0;
        color: #e2f6f2;
        font-size: 0.9rem;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] {
        display: flex;
        flex-direction: column;
        gap: 0.25rem;
        padding: 0.35rem 0;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        margin: 0;
        padding: 0.5rem 0.8rem;
        border: 1px solid transparent;
        border-radius: 10px;
        color: #45635f;
        cursor: pointer;
        transition: background 160ms ease, color 160ms ease, border-color 160ms ease;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
        border-color: #b9e2d7;
        background: #e6f6ef;
        color: #0f625a;
        font-weight: 700;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background: #f0f8f5;
    }
    div.stButton > button[kind="primary"] {
        border-radius: 10px;
        background: #0f766e;
        border-color: #0f766e;
    }
    .energy-card {
        --card-ink: #124c47;
        --card-edge: #8fcfba;
        --card-wash: #d8f2e7;
        position: relative;
        min-height: 112px;
        overflow: hidden;
        padding: 13px 15px 12px;
        border: 1px solid var(--card-edge);
        border-radius: 18px;
        background: linear-gradient(145deg, #ffffff 5%, var(--card-wash) 100%);
        box-shadow: 0 8px 22px rgba(27, 69, 67, 0.08);
        color: var(--card-ink);
        animation: card-arrive 560ms cubic-bezier(.2,.75,.25,1) both;
        transition: transform 220ms ease, box-shadow 220ms ease, border-color 220ms ease;
    }
    .energy-card::after {
        position: absolute;
        right: -30px;
        bottom: -55px;
        width: 125px;
        height: 125px;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.45);
        content: "";
        transition: transform 260ms ease;
    }
    .energy-card:hover {
        z-index: 1;
        transform: translateY(-6px) scale(1.015);
        border-color: var(--card-ink);
        box-shadow: 0 16px 30px rgba(27, 69, 67, 0.15);
    }
    .energy-card:hover::after {
        transform: scale(1.45);
    }
    .energy-card--blue {
        --card-ink: #164c86;
        --card-edge: #9dbbe9;
        --card-wash: #dceaff;
    }
    .energy-card--amber {
        --card-ink: #744000;
        --card-edge: #e8bf72;
        --card-wash: #ffedc5;
    }
    .energy-card--violet {
        --card-ink: #52318b;
        --card-edge: #c1a7ec;
        --card-wash: #e9ddff;
    }
    .energy-card--coral {
        --card-ink: #8a3028;
        --card-edge: #e6a69a;
        --card-wash: #ffe0d8;
    }
    .energy-card--compact {
        min-height: 86px;
        padding: 9px 12px;
        border-radius: 14px;
    }
    .energy-card--compact .energy-card__icon {
        width: 28px;
        height: 28px;
        flex-basis: 28px;
    }
    .energy-card--compact .energy-card__value {
        margin-top: 5px;
        font-size: 1.35rem;
    }
    .energy-card__top {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
    }
    .energy-card__label {
        color: #344f4b;
        font-size: 0.87rem;
        font-weight: 700;
        letter-spacing: 0.015em;
    }
    .energy-card__icon {
        display: grid;
        width: 32px;
        height: 32px;
        flex: 0 0 32px;
        place-items: center;
        border: 1px solid rgba(255,255,255,0.85);
        border-radius: 11px;
        background: rgba(255,255,255,0.72);
        font-size: 1rem;
        animation: icon-pop 650ms ease both;
    }
    .energy-card__value {
        position: relative;
        z-index: 1;
        margin-top: 10px;
        color: var(--card-ink);
        font-size: clamp(1.35rem, 2vw, 1.85rem);
        font-weight: 800;
        line-height: 1.15;
        letter-spacing: -0.035em;
        overflow-wrap: anywhere;
    }
    .chart-spacer {
        height: 1.75rem;
    }
    .workflow-list {
        display: grid;
        gap: 0.65rem;
        margin: 0.75rem 0 1rem;
    }
    .workflow-step {
        display: grid;
        grid-template-columns: 42px 1fr;
        align-items: start;
        gap: 0.8rem;
        padding: 0.85rem 1rem;
        border: 1px solid #c8e1da;
        border-radius: 14px;
        background: linear-gradient(100deg, #ffffff 0%, #edf8f4 100%);
        box-shadow: 0 4px 12px rgba(28, 67, 74, 0.05);
    }
    .workflow-step__number {
        display: grid;
        width: 36px;
        height: 36px;
        place-items: center;
        border-radius: 12px;
        background: #0f766e;
        color: #ffffff;
        font-weight: 800;
    }
    .workflow-step__title {
        margin: 0 0 0.15rem;
        color: #173f3a;
        font-weight: 800;
    }
    .workflow-step__description {
        margin: 0;
        color: #365955;
        line-height: 1.45;
    }
    @keyframes card-arrive {
        from { opacity: 0; transform: translateY(14px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes icon-pop {
        0% { opacity: 0; transform: scale(.7) rotate(-10deg); }
        70% { transform: scale(1.08) rotate(3deg); }
        100% { opacity: 1; transform: scale(1) rotate(0); }
    }
    @media (prefers-reduced-motion: reduce) {
        .energy-card, .energy-card__icon {
            animation: none;
            transition: none;
        }
    }
    @media (max-height: 760px) {
        .hero { padding: 0.5rem 1rem; }
        .hero h1 { font-size: 1.35rem; }
        .energy-card { min-height: 96px; padding: 10px 12px; }
        .energy-card__value { margin-top: 6px; font-size: 1.35rem; }
        [data-testid="stMainBlockContainer"] { gap: 0.35rem; }
    }
    @media (max-width: 767px) {
        [data-testid="stAppViewContainer"] {
            display: flex !important;
            position: relative !important;
            align-items: stretch;
        }
        [data-testid="stSidebar"] {
            position: relative !important;
            inset: auto !important;
            flex: 0 0 260px !important;
            width: 260px !important;
            min-width: 260px !important;
        }
        [data-testid="stMain"] {
            position: relative !important;
            flex: 1 1 0 !important;
            width: calc(100% - 260px) !important;
            min-width: 0;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Reading and preparing the household readings...")
def load_path_data(path: str, modified_ns: int) -> pd.Series:
    return load_household_data(path)


@st.cache_data(show_spinner="Reading and preparing the uploaded household readings...")
def load_uploaded_data(content: bytes) -> pd.Series:
    return load_household_data(BytesIO(content))


def series_signature(series: pd.Series) -> str:
    payload = pd.util.hash_pandas_object(series, index=True).to_numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def render_metric_cards(
    cards: list[tuple[str, str, str, str]],
    compact: bool = False,
) -> None:
    columns = st.columns(len(cards))
    for index, (label, value, icon, variant) in enumerate(cards):
        with columns[index]:
            compact_class = " energy-card--compact" if compact else ""
            st.markdown(
                f"""
                <div class="energy-card energy-card--{variant}{compact_class}"
                     style="animation-delay: {index * 90}ms"
                     role="group"
                     aria-label="{escape(label)}: {escape(value)}">
                  <div class="energy-card__top">
                    <span class="energy-card__label">{escape(label)}</span>
                    <span class="energy-card__icon" aria-hidden="true">{escape(icon)}</span>
                  </div>
                  <div class="energy-card__value">{escape(value)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def make_power_chart(
    x: list[pd.Timestamp] | pd.DatetimeIndex,
    y: list[float] | pd.Series,
    title: str,
    color: str,
) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            name="Active power",
            line={"color": color, "width": 4 if title.startswith("Next-hour") else 3},
            marker={
                "size": 13 if title.startswith("Next-hour") else 8,
                "color": color,
                "symbol": "circle",
                "line": {"color": "#ffffff", "width": 2.5},
            },
            connectgaps=False,
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br><b>%{y:.2f} kW</b><extra></extra>",
        )
    )
    figure.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Mean active power (kW)",
        margin={"l": 20, "r": 20, "t": 55, "b": 20},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={"color": "#172f2c", "size": 13},
        hovermode="x unified",
        xaxis={
            "showgrid": True,
            "gridcolor": "#e1e9e7",
            "tickfont": {"color": "#294845", "size": 12},
            "title_font": {"color": "#173f3a", "size": 13},
        },
        yaxis={
            "showgrid": True,
            "gridcolor": "#d6e1df",
            "tickfont": {"color": "#294845", "size": 12},
            "title_font": {"color": "#173f3a", "size": 13},
        },
    )
    return figure


st.markdown(
    """
    <div class="hero">
      <h1>⚡ Smart Home Energy Forecasting</h1>
      <p>Understand household energy patterns and explore an LSTM forecast for the next hour.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.title("⚡ Energy dashboard")
    st.caption("NAVIGATION")
    page = st.radio(
        "Navigate",
        ["Overview", "Forecast", "Usage analytics", "About the model"],
        label_visibility="collapsed",
    )
    st.divider()
    st.subheader("Data and settings")
    uploaded = st.file_uploader(
        "Upload the UCI household power text file",
        type=["txt"],
        help="Expected columns include Date, Time, and Global_active_power.",
    )
    threshold_kw = st.number_input(
        "High-consumption warning threshold (kW)",
        min_value=0.5,
        max_value=11.0,
        value=3.0,
        step=0.1,
    )
    st.caption("Input history: 96 readings (24 hours). Forecast: four 15-minute intervals.")

try:
    if uploaded is not None:
        series = load_uploaded_data(uploaded.getvalue())
        source_label = uploaded.name
    elif DEFAULT_DATA_PATH.is_file():
        modified_ns = DEFAULT_DATA_PATH.stat().st_mtime_ns
        series = load_path_data(str(DEFAULT_DATA_PATH), modified_ns)
        source_label = DEFAULT_DATA_PATH.name
    else:
        st.info(
            "Upload the UCI file to get started. The app also checks "
            f"`{DEFAULT_DATA_PATH}` automatically."
        )
        st.stop()
except (OSError, ValueError, UnicodeDecodeError, pd.errors.ParserError) as exc:
    st.error(f"Could not read the selected dataset: {exc}")
    st.stop()

if len(series) <= LOOKBACK_STEPS + FORECAST_STEPS:
    st.error("The dataset is too short. Provide at least 25 hours of readings.")
    st.stop()

signature = series_signature(series)
if st.session_state.get("model_signature") != signature:
    st.session_state.pop("model", None)
    st.session_state.pop("model_signature", None)

latest_time = series.index[-1]
current_kw = float(series.iloc[-1]) if pd.notna(series.iloc[-1]) else None
daily_energy = series.resample("D").sum(min_count=1) * 0.25
today_energy = daily_energy.loc[latest_time.normalize()]
today_peak_kw = float(series.loc[latest_time.normalize() :].max())
model_state = st.session_state.get("model")
future_kw = None
validation_rmse = None

if model_state is not None and series.iloc[-LOOKBACK_STEPS:].notna().all():
    model, mean, scale, validation_rmse = model_state
    future_kw = forecast_next_hour(
        model,
        series.iloc[-LOOKBACK_STEPS:].to_numpy(),
        mean,
        scale,
    )

st.caption(
    f"Source: {source_label} · Latest dataset reading: {latest_time:%Y-%m-%d %H:%M} · "
    f"{len(series):,} 15-minute intervals"
)

if page == "Overview":
    st.subheader("At a glance")
    render_metric_cards(
        [
            (
                "Latest consumption",
                f"{current_kw:.2f} kW" if current_kw is not None else "Unavailable",
                "⚡",
                "mint",
            ),
            ("Energy on latest date", f"{today_energy:.2f} kWh", "🔋", "blue"),
            ("Peak on latest date", f"{today_peak_kw:.2f} kW", "📈", "amber"),
            (
                "Predicted in 15 minutes",
                f"{future_kw[0]:.2f} kW" if future_kw is not None else "Train model",
                "✨",
                "violet",
            ),
        ]
    )
    st.caption("Forecast value is for the next interval after the latest dataset reading.")

    if current_kw is None:
        st.warning("The latest interval has no usable active-power reading.")
    elif current_kw >= threshold_kw:
        st.warning(
            f"High-consumption warning: {current_kw:.2f} kW is above your "
            f"{threshold_kw:.1f} kW threshold."
        )
        st.info(
            "Energy-saving idea: check for appliances running at the same time "
            "and shift flexible loads away from this peak where practical."
        )
    else:
        st.success("Latest use is below your selected high-consumption threshold.")
        st.info(
            "Energy-saving idea: track daily patterns and schedule flexible "
            "appliances during lower-use periods."
        )

    st.caption(
        "The dataset is historical, so the latest reading is the last timestamp in "
        "the file, not a live smart-meter reading. Open Usage analytics for the charts."
    )

elif page == "Forecast":
    st.subheader("Next-hour forecast")
    st.caption(
        "The LSTM uses the 96 15-minute readings from the previous 24 hours to "
        "estimate the next four 15-minute average power values."
    )
    train_clicked = st.button(
        "Train / refresh LSTM",
        type="primary",
        help="Trains on up to 180 recent days and reserves the final 20% for validation.",
    )
    if train_clicked:
        try:
            with st.spinner("Training the LSTM on recent household readings..."):
                trained_model, mean, scale, validation_rmse = train_lstm(series)
            st.session_state["model"] = (
                trained_model,
                mean,
                scale,
                validation_rmse,
            )
            st.session_state["model_signature"] = signature
            st.rerun()
        except (RuntimeError, ValueError) as exc:
            st.error(f"Could not train the forecast model: {exc}")

    if future_kw is None:
        if model_state is not None:
            st.warning(
                "The latest 24 hours contain a long data gap. A forecast cannot "
                "be generated until there is a complete 24-hour input window."
            )
        else:
            st.info("Train the LSTM to see the four forecast values and forecast chart.")
    else:
        render_metric_cards(
            [
                (
                    f"{(step + 1) * 15} min ahead",
                    f"{future_kw[step]:.2f} kW",
                    ("⏱️", "🔮", "📊", "⚡")[step],
                    ("mint", "blue", "amber", "violet")[step],
                )
                for step in range(FORECAST_STEPS)
            ],
            compact=True,
        )

        st.markdown('<div class="chart-spacer"></div>', unsafe_allow_html=True)
        future_times = [
            latest_time + pd.Timedelta(minutes=15 * step)
            for step in range(1, FORECAST_STEPS + 1)
        ]
        if current_kw is not None:
            chart_times = [latest_time, *future_times]
            chart_values = [current_kw, *future_kw]
        else:
            chart_times = future_times
            chart_values = list(future_kw)
        st.plotly_chart(
            make_power_chart(
                chart_times,
                chart_values,
                "Next-hour power forecast",
                ACCENT,
            ),
            use_container_width=True,
            height=420,
        )
        st.caption(f"Validation RMSE: {validation_rmse:.3f} kW · Held-out data; future accuracy is not guaranteed.")

elif page == "Usage analytics":
    st.subheader("Usage analytics")
    render_metric_cards(
        [
            ("Energy on latest date", f"{today_energy:.2f} kWh", "🔋", "blue"),
            ("Peak on latest date", f"{today_peak_kw:.2f} kW", "📈", "amber"),
            ("Average power on latest date", f"{today_energy / 24:.2f} kW", "⚡", "mint"),
        ]
    )
    st.markdown('<div class="chart-spacer"></div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        recent = series.tail(7 * LOOKBACK_STEPS)
        st.plotly_chart(
            make_power_chart(recent.index, recent, "Recent power history", "#3977A8"),
            use_container_width=True,
            height=350,
        )
    with right:
        recent_daily = daily_energy.tail(14)
        daily_chart = go.Figure(
            go.Bar(
                x=recent_daily.index,
                y=recent_daily,
                name="Energy",
                marker={
                    "color": "#E9A23B",
                    "line": {"color": "#9B5B0B", "width": 1.5},
                },
                texttemplate="%{y:.1f}",
                textposition="outside",
                textfont={"color": "#744000", "size": 11},
                cliponaxis=False,
                hovertemplate="%{x|%Y-%m-%d}<br><b>%{y:.2f} kWh</b><extra></extra>",
            )
        )
        daily_chart.update_layout(
            title="Daily energy — latest 14 dates",
            xaxis_title="Date",
            yaxis_title="Energy (kWh)",
            margin={"l": 20, "r": 20, "t": 55, "b": 20},
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font={"color": "#172f2c", "size": 12},
            xaxis={
                "showgrid": False,
                "tickfont": {"color": "#294845", "size": 10},
                "title_font": {"color": "#173f3a"},
            },
            yaxis={
                "showgrid": True,
                "gridcolor": "#d6e1df",
                "tickfont": {"color": "#294845", "size": 11},
                "title_font": {"color": "#173f3a"},
                "rangemode": "tozero",
            },
        )
        st.plotly_chart(daily_chart, use_container_width=True, height=350)

elif page == "About the model":
    st.subheader("How the forecast works")
    st.caption(
        "A clear view of how raw household readings become the next-hour estimate."
    )
    workflow_steps = [
        (
            "Load readings",
            "Read the UCI timestamps and Global_active_power values from the selected text file.",
        ),
        (
            "Prepare intervals",
            "Average minute readings into 15-minute kW values; interpolate only short gaps.",
        ),
        (
            "Build sequences",
            "Use the previous 96 intervals (24 hours) as input and the next four intervals as targets.",
        ),
        (
            "Train and validate",
            "Train a 32-unit LSTM on up to 180 recent days; hold out the final 20% for validation.",
        ),
        (
            "Forecast the next hour",
            "Predict the 15-, 30-, 45-, and 60-minute average power values and display them in kW.",
        ),
    ]
    workflow_html = "".join(
        (
            '<div class="workflow-step">'
            f'<div class="workflow-step__number">{index}</div>'
            "<div>"
            f'<p class="workflow-step__title">{escape(title)}</p>'
            f'<p class="workflow-step__description">{escape(description)}</p>'
            "</div>"
            "</div>"
        )
        for index, (title, description) in enumerate(workflow_steps, start=1)
    )
    st.markdown(
        f'<div class="workflow-list">{workflow_html}</div>',
        unsafe_allow_html=True,
    )
    st.info(
        "Daily energy is estimated by summing each 15-minute mean power reading "
        "multiplied by 0.25 hours. Short missing-data gaps are interpolated; long "
        "gaps are retained and excluded from training sequences."
    )
    st.warning(
        "This dashboard analyzes the historical dataset. It does not connect to "
        "a live household meter, and its predictions are estimates rather than "
        "guaranteed readings."
    )
