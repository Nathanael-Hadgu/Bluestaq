import json
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(page_title="Straight Rows SDA Explorer", layout="wide")
st.title("🚜 Straight Rows: Telemetry & SDA Explorer")


@st.cache_data
def load_telemetry():
    path = "data/farm/tractor_telemetry.csv.gz"
    df = pd.read_csv(path)
    df["ts"] = pd.to_datetime(df["ts"])
    return df


@st.cache_data
def load_labels():
    path = "data/labels/train_episodes.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


with st.spinner("Loading 777,600 rows of tractor telemetry..."):
    df = load_telemetry()
    labels = load_labels()

st.sidebar.header("Filter Controls")

view_layout = st.sidebar.radio(
    "Graph Layout:",
    [
        "Individual Graphs (Tabs)",
        "Individual Graphs (Stacked Full-Size)",
        "Synchronized Subplots",
    ],
)

window_mode = st.sidebar.radio(
    "Select Time Window:",
    [
        "Development (May 4 - May 6)",
        "Evaluation / Scored (May 6 - May 7)",
        "Full 72 Hours",
    ],
)

if window_mode == "Development (May 4 - May 6)":
    start_bound = pd.to_datetime("2026-05-04 00:00:00+00:00")
    end_bound = pd.to_datetime("2026-05-06 00:00:00+00:00")
elif window_mode == "Evaluation / Scored (May 6 - May 7)":
    start_bound = pd.to_datetime("2026-05-06 00:00:00+00:00")
    end_bound = pd.to_datetime("2026-05-07 00:00:00+00:00")
else:
    start_bound = df["ts"].min()
    end_bound = df["ts"].max()

selected_tractors = st.sidebar.multiselect(
    "Tractors:", options=["T1", "T2", "T3"], default=["T1", "T2", "T3"]
)

resample_rate = st.sidebar.selectbox(
    "Downsample Rate (use 1s when zoomed into a short window):",
    options=["10s", "30s", "1min", "1s (Raw 1Hz)"],
    index=0,
)

only_working = st.sidebar.checkbox(
    "Only show working tractors (speed > 0)", value=False
)

mask = (
    (df["ts"] >= start_bound)
    & (df["ts"] <= end_bound)
    & (df["tractor_id"].isin(selected_tractors))
)
sub_df = df.loc[mask].copy()

if only_working:
    sub_df = sub_df[sub_df["speed_mps"] > 0.1]

if resample_rate != "1s (Raw 1Hz)":
    sub_df = (
        sub_df.set_index("ts")
        .groupby("tractor_id")
        .resample(resample_rate)
        .agg({
            "cross_track_error_m": "max",
            "mean_cn0_dbhz": "mean",
            "min_cn0_dbhz": "min",
            "num_sats": "min",
            "speed_mps": "mean",
            "base_link_ok": "min",
            "lat": "first",
            "lon": "first",
            "fix_state": "first",
        })
        .reset_index()
    )

colors = {"T1": "#EF553B", "T2": "#00CC96", "T3": "#636EFA"}


def make_single_chart(data, y_col, title, y_label):
    fig = px.line(
        data,
        x="ts",
        y=y_col,
        color="tractor_id",
        color_discrete_map=colors,
        title=title,
        labels={"ts": "Timestamp (UTC)", y_col: y_label, "tractor_id": "Tractor"},
        hover_data=["fix_state", "speed_mps"],
    )
    fig.update_layout(height=450, hovermode="x unified", template="plotly_dark")
    return fig


st.subheader("Fleet Telemetry")
sub_df["base_link_num"] = sub_df["base_link_ok"].astype(float)

if view_layout == "Individual Graphs (Tabs)":
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📏 Cross-Track Error",
        "📶 Mean C/N0",
        "🔽 Min C/N0",
        "🛰️ Satellite Count",
        "🔗 Base Link & Speed",
    ])
    with tab1:
        st.plotly_chart(
            make_single_chart(sub_df, "cross_track_error_m", "Cross-Track Error (m)", "Error (m)"),
            use_container_width=True,
        )
    with tab2:
        st.plotly_chart(
            make_single_chart(sub_df, "mean_cn0_dbhz", "Mean C/N0 Signal Strength (dB-Hz)", "dB-Hz"),
            use_container_width=True,
        )
    with tab3:
        st.plotly_chart(
            make_single_chart(sub_df, "min_cn0_dbhz", "Minimum C/N0 Signal Strength (dB-Hz)", "dB-Hz"),
            use_container_width=True,
        )
    with tab4:
        st.plotly_chart(
            make_single_chart(sub_df, "num_sats", "Satellite Count", "Satellites"),
            use_container_width=True,
        )
    with tab5:
        st.plotly_chart(
            make_single_chart(sub_df, "base_link_num", "Base Station Link Status (1=OK, 0=Lost)", "Link Status"),
            use_container_width=True,
        )
        st.plotly_chart(
            make_single_chart(sub_df, "speed_mps", "Tractor Speed (m/s)", "Speed (m/s)"),
            use_container_width=True,
        )

elif view_layout == "Individual Graphs (Stacked Full-Size)":
    st.plotly_chart(make_single_chart(sub_df, "cross_track_error_m", "Cross-Track Error (m)", "Error (m)"), use_container_width=True)
    st.plotly_chart(make_single_chart(sub_df, "mean_cn0_dbhz", "Mean C/N0 Signal Strength (dB-Hz)", "dB-Hz"), use_container_width=True)
    st.plotly_chart(make_single_chart(sub_df, "min_cn0_dbhz", "Minimum C/N0 Signal Strength (dB-Hz)", "dB-Hz"), use_container_width=True)
    st.plotly_chart(make_single_chart(sub_df, "num_sats", "Satellite Count", "Satellites"), use_container_width=True)
    st.plotly_chart(make_single_chart(sub_df, "base_link_num", "Base Station Link Status (1=OK, 0=Lost)", "Link Status"), use_container_width=True)

else:
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=(
            "Cross-Track Error (m) [Max]",
            "Mean C/N0 Signal Strength (dB-Hz)",
            "Satellite Count (num_sats)",
            "Base Station Link OK (1=True, 0=False)",
        ),
    )
    for tid in selected_tractors:
        t_df = sub_df[sub_df["tractor_id"] == tid]
        c = colors.get(tid, "#AB63FA")
        fig.add_trace(go.Scattergl(x=t_df["ts"], y=t_df["cross_track_error_m"], name=f"{tid} XTE", line=dict(color=c), legendgroup=tid), row=1, col=1)
        fig.add_trace(go.Scattergl(x=t_df["ts"], y=t_df["mean_cn0_dbhz"], name=f"{tid} C/N0", line=dict(color=c), legendgroup=tid, showlegend=False), row=2, col=1)
        fig.add_trace(go.Scattergl(x=t_df["ts"], y=t_df["num_sats"], name=f"{tid} Sats", line=dict(color=c), legendgroup=tid, showlegend=False), row=3, col=1)
        fig.add_trace(go.Scattergl(x=t_df["ts"], y=t_df["base_link_num"], name=f"{tid} BaseLink", line=dict(color=c), legendgroup=tid, showlegend=False), row=4, col=1)
    fig.update_layout(height=800, hovermode="x unified", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("🗺️ Spatial Field Map (Colored by Cross-Track Error)")
map_df = sub_df.dropna(subset=["lat", "lon", "cross_track_error_m"])

if hasattr(px, "scatter_map"):
    fig_map = px.scatter_map(
        map_df,
        lat="lat",
        lon="lon",
        color="cross_track_error_m",
        hover_data=["ts", "tractor_id", "fix_state", "mean_cn0_dbhz", "num_sats"],
        color_continuous_scale="Turbo",
        range_color=[0, 1.0],
        zoom=14,
        height=550,
    )
    fig_map.update_layout(map_style="open-street-map")
else:
    fig_map = px.scatter(
        map_df,
        x="lon",
        y="lat",
        color="cross_track_error_m",
        hover_data=["ts", "tractor_id", "fix_state", "mean_cn0_dbhz", "num_sats"],
        color_continuous_scale="Turbo",
        range_color=[0, 1.0],
        height=550,
    )

st.plotly_chart(fig_map, use_container_width=True)

if labels is not None and window_mode != "Evaluation / Scored (May 6 - May 7)":
    st.subheader("📋 Labeled Development Episodes (Ground Truth)")
    st.json(labels)
EOF