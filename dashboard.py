import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(page_title="Straight Rows: GNSS & SDA Attribution Console", layout="wide")
st.title("🚜 Straight Rows: Precision Guidance & SDA Attribution Console")
st.caption("Multi-Label Farm Telemetry + Space Domain Awareness (SDA) Fusion Decision Support System")


@st.cache_data
def load_all_data():
    data_dir = Path("data")
    df = pd.read_csv(data_dir / "farm/tractor_telemetry.csv.gz")
    df["ts"] = pd.to_datetime(df["ts"], utc=True)

    with open(data_dir / "sda/sgi_indices.json") as f:
        sgi = json.load(f)
    kp_df = pd.DataFrame(sgi["kp"])
    kp_df["ts"] = pd.to_datetime(kp_df["ts"], utc=True)
    dst_df = pd.DataFrame(sgi["dst"])
    dst_df["ts"] = pd.to_datetime(dst_df["ts"], utc=True)

    with open(data_dir / "sda/scintillation_series.json") as f:
        s4_df = pd.DataFrame(json.load(f)["series"])
    s4_df["ts"] = pd.to_datetime(s4_df["ts"], utc=True)

    with open(data_dir / "sda/constellation_advisories.json") as f:
        advisories = json.load(f)["advisories"]

    with open(data_dir / "sda/rfi_reports.json") as f:
        rfi_reports = json.load(f)["reports"]

    train_labels = None
    if (data_dir / "labels/train_episodes.json").exists():
        with open(data_dir / "labels/train_episodes.json") as f:
            train_labels = json.load(f)["episodes"]

    sub_dets = []
    if Path("submission.json").exists():
        with open("submission.json") as f:
            sub_dets = json.load(f).get("detections", [])

    return df, kp_df, dst_df, s4_df, advisories, rfi_reports, train_labels, sub_dets


with st.spinner("Loading 777,600 rows of 1Hz farm telemetry and SDA feeds..."):
    df, kp_df, dst_df, s4_df, advisories, rfi_reports, train_labels, sub_dets = load_all_data()

# Sidebar Controls
st.sidebar.header("🎛️ Operator Controls")

window_mode = st.sidebar.radio(
    "Select Time Window:",
    [
        "Evaluation / Scored (May 6 - May 7)",
        "Development (May 4 - May 6)",
        "Full 72 Hours",
    ],
    index=0,
)

if window_mode == "Development (May 4 - May 6)":
    start_bound = pd.to_datetime("2026-05-04 00:00:00+00:00")
    end_bound = pd.to_datetime("2026-05-06 00:00:00+00:00")
    active_episodes = train_labels or []
elif window_mode == "Evaluation / Scored (May 6 - May 7)":
    start_bound = pd.to_datetime("2026-05-06 00:00:00+00:00")
    end_bound = pd.to_datetime("2026-05-07 00:00:00+00:00")
    active_episodes = sub_dets
else:
    start_bound = df["ts"].min()
    end_bound = df["ts"].max()
    active_episodes = (train_labels or []) + sub_dets

# Quick Episode Jump Filter
macro_eps = [e for e in active_episodes if e.get("class") != "MULTIPATH"]
ep_options = ["Full Selected Window"] + [
    f"{i+1}. {e['class']} ({','.join(e['tractors'])}) [{e['start'][11:16]} - {e['end'][11:16]}Z]"
    for i, e in enumerate(macro_eps)
]
selected_ep_label = st.sidebar.selectbox("🔍 Zoom to Attributed Episode:", ep_options, index=0)

if selected_ep_label != "Full Selected Window":
    ep_idx = int(selected_ep_label.split(".")[0]) - 1
    chosen_ep = macro_eps[ep_idx]
    start_bound = pd.to_datetime(chosen_ep["start"]) - pd.Timedelta(minutes=10)
    end_bound = pd.to_datetime(chosen_ep["end"]) + pd.Timedelta(minutes=10)
else:
    chosen_ep = None

view_layout = st.sidebar.radio(
    "Graph Layout:",
    [
        "Individual Graphs (Tabs)",
        "Synchronized Subplots (Telemetry + SDA)",
        "Individual Graphs (Stacked Full-Size)",
    ],
)

selected_tractors = st.sidebar.multiselect(
    "Tractors:", options=["T1", "T2", "T3"], default=["T1", "T2", "T3"]
)

default_resample = 3 if chosen_ep is not None else 0
resample_rate = st.sidebar.selectbox(
    "Downsample Rate (use 1s when zoomed into a short window):",
    options=["10s", "30s", "1min", "1s (Raw 1Hz)"],
    index=default_resample,
)

only_working = st.sidebar.checkbox("Only show working tractors (speed > 0)", value=False)
show_ep_bands = st.sidebar.checkbox("Highlight detected macro episodes on charts", value=True)

# Top KPI Summary Banner
macro_count = len([e for e in active_episodes if e.get("class") != "MULTIPATH"])
mp_count = len([e for e in active_episodes if e.get("class") == "MULTIPATH"])
c1, c2, c3, c4 = st.columns(4)
c1.metric("Macro Guidance Episodes", f"{macro_count}")
c2.metric("Headland Multipath Passes (<=3m)", f"{mp_count}")
c3.metric("Regional S4 Scintillation Peak", f"{s4_df[(s4_df.ts >= start_bound) & (s4_df.ts <= end_bound)]['s4'].max():.3f}")
c4.metric("Planetary Kp Peak", f"{kp_df[(kp_df.ts >= start_bound) & (kp_df.ts <= end_bound)]['kp'].max():.1f}")

# If an episode is selected, show Operator Decision & Evidence Card at the top
if chosen_ep is not None:
    action_val = chosen_ep.get("action") or chosen_ep.get("recommended_action", "")
    conf_val = chosen_ep.get("confidence", 1.0)
    st.info(
        f"### 🚨 Active Attribution: `{chosen_ep['class']}` on `{', '.join(chosen_ep['tractors'])}`\n"
        f"**Window:** `{chosen_ep['start']}` → `{chosen_ep['end']}` | "
        f"**Recommended Action:** `{action_val}` | **Calibrated Confidence:** `{conf_val:.2f}`\n\n"
        + "\n".join([f"- {ev_line}" for ev_line in chosen_ep.get("evidence", ["Ground-truth labeled development episode."])])
    )

# Filter DataFrame
mask = (
    (df["ts"] >= start_bound)
    & (df["ts"] <= end_bound)
    & (df["tractor_id"].isin(selected_tractors))
)
sub_df = df.loc[mask].copy()

if only_working:
    sub_df = sub_df[sub_df["speed_mps"] > 0.1]

if resample_rate != "1s (Raw 1Hz)" and not sub_df.empty:
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

sub_df["base_link_num"] = sub_df["base_link_ok"].astype(float)
colors = {"T1": "#EF553B", "T2": "#00CC96", "T3": "#636EFA"}
class_colors = {
    "SPACE_WEATHER": "rgba(255, 165, 0, 0.18)",
    "JAMMING": "rgba(255, 50, 50, 0.20)",
    "SPOOFING": "rgba(180, 50, 255, 0.20)",
    "EQUIPMENT_FAULT": "rgba(0, 200, 255, 0.18)",
}


def add_episode_bands(fig):
    if not show_ep_bands:
        return
    for ep in macro_eps:
        es, ee = pd.to_datetime(ep["start"]), pd.to_datetime(ep["end"])
        if ee >= start_bound and es <= end_bound:
            fig.add_vrect(
                x0=max(es, start_bound),
                x1=min(ee, end_bound),
                fillcolor=class_colors.get(ep["class"], "rgba(200,200,200,0.15)"),
                layer="below",
                line_width=1,
                annotation_text=f"{ep['class']} ({','.join(ep['tractors'])})",
                annotation_position="top left",
            )


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
    add_episode_bands(fig)
    fig.update_layout(height=450, hovermode="x unified", template="plotly_dark")
    return fig


def make_sda_chart():
    s4_sub = s4_df[(s4_df.ts >= start_bound) & (s4_df.ts <= end_bound)]
    kp_sub = kp_df[(kp_df.ts >= start_bound - pd.Timedelta(hours=3)) & (kp_df.ts <= end_bound)]
    dst_sub = dst_df[(dst_df.ts >= start_bound) & (dst_df.ts <= end_bound)]
    fig_sda = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=("Regional Scintillation Proxy (S4 Index) & 3-Hourly Kp", "Geomagnetic Storm Dst Index (nT)")
    )
    fig_sda.add_trace(go.Scatter(x=s4_sub["ts"], y=s4_sub["s4"], name="S4 Index (x10 scaled)", line=dict(color="#00CC96", width=2)), row=1, col=1)
    fig_sda.add_trace(go.Scatter(x=kp_sub["ts"], y=kp_sub["kp"] / 10.0, name="Kp / 10", line=dict(color="#FFA15A", width=2, shape="hv")), row=1, col=1)
    fig_sda.add_hline(y=0.30, line_dash="dash", line_color="red", annotation_text="S4 Storm Threshold (0.30)", row=1, col=1)
    fig_sda.add_trace(go.Scatter(x=dst_sub["ts"], y=dst_sub["dst_nT"], name="Dst (nT)", line=dict(color="#AB63FA", width=2)), row=2, col=1)
    fig_sda.update_layout(height=480, hovermode="x unified", template="plotly_dark")
    return fig_sda


st.subheader("📈 Fleet Telemetry & Space Domain Awareness (SDA) Feeds")

if view_layout == "Individual Graphs (Tabs)":
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📏 Cross-Track Error",
        "📶 Mean C/N0",
        "🔽 Min C/N0 (Antenna/Multipath)",
        "🛰️ Satellite Count",
        "🔗 Base Link & Speed",
        "☀️ SDA Feeds (S4, Kp, Dst, RFI)",
    ])
    with tab1:
        st.plotly_chart(make_single_chart(sub_df, "cross_track_error_m", "Cross-Track Error (m)", "Error (m)"), use_container_width=True)
    with tab2:
        st.plotly_chart(make_single_chart(sub_df, "mean_cn0_dbhz", "Mean C/N0 Signal Strength (dB-Hz)", "dB-Hz"), use_container_width=True)
    with tab3:
        st.plotly_chart(make_single_chart(sub_df, "min_cn0_dbhz", "Minimum C/N0 Signal Strength (dB-Hz)", "dB-Hz"), use_container_width=True)
    with tab4:
        st.plotly_chart(make_single_chart(sub_df, "num_sats", "Satellite Count", "Satellites"), use_container_width=True)
    with tab5:
        st.plotly_chart(make_single_chart(sub_df, "base_link_num", "Base Station Link Status (1=OK, 0=Lost)", "Link Status"), use_container_width=True)
        st.plotly_chart(make_single_chart(sub_df, "speed_mps", "Tractor Speed (m/s)", "Speed (m/s)"), use_container_width=True)
    with tab6:
        st.plotly_chart(make_sda_chart(), use_container_width=True)
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**🛰️ Constellation Advisories (`constellation_advisories.json`)**")
            st.dataframe(pd.DataFrame(advisories), use_container_width=True)
        with col_b:
            st.markdown("**📡 Regional RFI Reports (`rfi_reports.json`)**")
            st.dataframe(pd.DataFrame(rfi_reports), use_container_width=True)

elif view_layout == "Individual Graphs (Stacked Full-Size)":
    st.plotly_chart(make_single_chart(sub_df, "cross_track_error_m", "Cross-Track Error (m)", "Error (m)"), use_container_width=True)
    st.plotly_chart(make_single_chart(sub_df, "mean_cn0_dbhz", "Mean C/N0 Signal Strength (dB-Hz)", "dB-Hz"), use_container_width=True)
    st.plotly_chart(make_single_chart(sub_df, "min_cn0_dbhz", "Minimum C/N0 Signal Strength (dB-Hz)", "dB-Hz"), use_container_width=True)
    st.plotly_chart(make_single_chart(sub_df, "num_sats", "Satellite Count", "Satellites"), use_container_width=True)
    st.plotly_chart(make_single_chart(sub_df, "base_link_num", "Base Station Link Status (1=OK, 0=Lost)", "Link Status"), use_container_width=True)
    st.plotly_chart(make_sda_chart(), use_container_width=True)

else:
    s4_sub = s4_df[(s4_df.ts >= start_bound) & (s4_df.ts <= end_bound)]
    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(
            "Cross-Track Error (m) [Max]",
            "Mean C/N0 Signal Strength (dB-Hz)",
            "Minimum C/N0 (dB-Hz) & Satellite Count",
            "Base Station Link OK (1=True, 0=False)",
            "SDA Regional Scintillation S4 Index",
        ),
    )
    for tid in selected_tractors:
        t_df = sub_df[sub_df["tractor_id"] == tid]
        c = colors.get(tid, "#AB63FA")
        fig.add_trace(go.Scattergl(x=t_df["ts"], y=t_df["cross_track_error_m"], name=f"{tid} XTE", line=dict(color=c), legendgroup=tid), row=1, col=1)
        fig.add_trace(go.Scattergl(x=t_df["ts"], y=t_df["mean_cn0_dbhz"], name=f"{tid} C/N0", line=dict(color=c), legendgroup=tid, showlegend=False), row=2, col=1)
        fig.add_trace(go.Scattergl(x=t_df["ts"], y=t_df["min_cn0_dbhz"], name=f"{tid} Min C/N0", line=dict(color=c), legendgroup=tid, showlegend=False), row=3, col=1)
        fig.add_trace(go.Scattergl(x=t_df["ts"], y=t_df["base_link_num"], name=f"{tid} BaseLink", line=dict(color=c), legendgroup=tid, showlegend=False), row=4, col=1)
    fig.add_trace(go.Scatter(x=s4_sub["ts"], y=s4_sub["s4"], name="S4 Index", line=dict(color="#FFA15A", width=2)), row=5, col=1)
    fig.update_layout(height=950, hovermode="x unified", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

# Spatial Field Map + Northern Headland Multipath Boundary
st.subheader("🗺️ Spatial Field Map (Colored by Cross-Track Error — Northern Headland Multipath Zone at Lat >= 46.90685)")
map_df = sub_df.dropna(subset=["lat", "lon", "cross_track_error_m"])

if hasattr(px, "scatter_map"):
    fig_map = px.scatter_map(
        map_df,
        lat="lat",
        lon="lon",
        color="cross_track_error_m",
        hover_data=["ts", "tractor_id", "fix_state", "mean_cn0_dbhz", "min_cn0_dbhz", "num_sats"],
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
        hover_data=["ts", "tractor_id", "fix_state", "mean_cn0_dbhz", "min_cn0_dbhz", "num_sats"],
        color_continuous_scale="Turbo",
        range_color=[0, 1.0],
        height=550,
    )

st.plotly_chart(fig_map, use_container_width=True)

# Operator Detections & Evidence Table
st.subheader("🛡️ Attributed Guidance Detections & Operator Action Playbook")
hide_mp_table = st.checkbox("Show only Macro Episodes in table (hide 17-second recurrent headland multipath passes)", value=True)
display_eps = macro_eps if hide_mp_table else active_episodes

if display_eps:
    rows = []
    for ep in display_eps:
        rows.append({
            "Start (UTC)": ep["start"],
            "End (UTC)": ep["end"],
            "Class": ep["class"],
            "Tractors": ", ".join(ep["tractors"]),
            "Recommended Action": ep.get("action") or ep.get("recommended_action", ""),
            "Confidence": ep.get("confidence", 1.0),
            "Primary Evidence": ep.get("evidence", ["Ground truth training episode"])[0],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    with st.expander("📄 View Full Multi-Source Attribution Evidence JSON"):
        st.json(display_eps)
