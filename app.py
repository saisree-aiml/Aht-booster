"""
AHT BOOSTER — Intelligent Agent Recommendation & AHT Optimization
A Streamlit POC that reads an uploaded Excel file (source of truth) and
recommends the best agent to handle a given call category, with a
transparent, re-computed-every-time scoring engine and dynamic time-saving
calculations. Nothing about agents, scores, or savings is hardcoded.
"""

import io
import pandas as pd
import numpy as np
import streamlit as st

# --------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="AHT Booster",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

REQUIRED_COLUMNS = ["Agent ID", "Agent Name", "Call Category", "Average AHT (sec)"]
OPTIONAL_NUMERIC_COLUMNS = [
    "Calls Handled", "Target AHT (sec)", "Quality Score (%)",
    "FCR (%)", "Transfer Rate (%)", "Availability (%)", "Experience (Months)",
]

# Scoring weights (transparent, POC-level, editable here only)
W_AHT = 0.60
W_QUALITY = 0.15
W_FCR = 0.15
W_TRANSFER = 0.10
AVAILABILITY_THRESHOLD = 80.0

# --------------------------------------------------------------------------
# STYLE
# --------------------------------------------------------------------------
st.markdown("""
<style>
    .main .block-container {padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1100px;}
    #MainMenu, footer, header {visibility: hidden;}

    .app-header {
        background: linear-gradient(135deg, #0f2440 0%, #1b3a63 55%, #2563a8 100%);
        padding: 2rem 1.75rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .app-header h1 {
        margin: 0; font-size: 2.1rem; font-weight: 800; letter-spacing: 0.5px;
        color: #ffffff;
    }
    .app-header .subtitle {
        font-size: 1.02rem; font-weight: 500; color: #cfe0f5; margin-top: 0.35rem;
    }
    .app-header .desc {
        font-size: 0.92rem; color: #a9c3e0; margin-top: 0.6rem;
    }

    .section-title {
        font-size: 1.05rem; font-weight: 700; color: #0f2440;
        margin: 1.6rem 0 0.6rem 0; padding-bottom: 0.35rem;
        border-bottom: 2px solid #e3e9f2;
    }

    .kpi-card {
        background: white; border-radius: 14px; padding: 1.1rem 1rem;
        border: 1px solid #e3e9f2; box-shadow: 0 2px 10px rgba(15,36,64,0.05);
        text-align: center; height: 100%;
    }
    .kpi-label {
        font-size: 0.72rem; font-weight: 700; color: #6b7c93;
        text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 0.4rem;
    }
    .kpi-value {
        font-size: 1.65rem; font-weight: 800; color: #0f2440; line-height: 1.1;
    }
    .kpi-value.highlight { color: #1c8a4c; }
    .kpi-sub { font-size: 0.75rem; color: #8a97a8; margin-top: 0.25rem; }

    .impact-banner {
        background: linear-gradient(135deg, #eafaf1 0%, #dcf5e6 100%);
        border: 1px solid #b7ecc9; border-radius: 14px;
        padding: 1rem 1.3rem; margin: 1rem 0 1.4rem 0;
        font-size: 1.0rem; color: #0f5132; font-weight: 500;
    }

    .best-agent-card {
        background: linear-gradient(135deg, #fffdf5 0%, #fff9e6 100%);
        border: 1.5px solid #f2d98a; border-radius: 16px;
        padding: 1.5rem 1.6rem; margin-top: 0.5rem;
    }
    .best-agent-title { font-size: 0.8rem; font-weight: 700; color: #a3760a;
        text-transform: uppercase; letter-spacing: 0.8px; }
    .best-agent-name { font-size: 1.55rem; font-weight: 800; color: #0f2440; margin-top: 0.2rem;}
    .best-agent-id { font-size: 0.85rem; color: #6b7c93; font-weight: 600;}

    .metric-row { display:flex; justify-content: space-between; margin: 0.55rem 0 0.15rem 0;
        font-size: 0.85rem; color: #33465e; font-weight: 600;}

    .footnote { font-size: 0.78rem; color: #8a97a8; margin-top: 1.8rem;
        border-top: 1px solid #e3e9f2; padding-top: 0.8rem; }

    div[data-testid="stMetricValue"] { font-size: 1.4rem; }
    .stButton>button {
        background: linear-gradient(135deg, #1b3a63 0%, #2563a8 100%);
        color: white; font-weight: 700; font-size: 1.05rem;
        padding: 0.7rem 1.5rem; border-radius: 10px; border: none; width: 100%;
    }
    .stButton>button:hover { background: linear-gradient(135deg, #14304f 0%, #1e5390 100%); color: white; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# HEADER
# --------------------------------------------------------------------------
st.markdown("""
<div class="app-header">
    <h1>🎯 AHT BOOSTER</h1>
    <div class="subtitle">Intelligent Agent Recommendation &amp; AHT Optimization</div>
    <div class="desc">Route calls to the right agent and reduce average handle time.</div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------------
def validate_dataframe(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return False, f"The uploaded file is missing required column(s): {', '.join(missing)}"
    if df.empty:
        return False, "The uploaded file has no data rows."
    df = df.dropna(subset=REQUIRED_COLUMNS)
    if df.empty:
        return False, "No valid rows remain after removing rows with missing required fields."
    return True, ""


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = ["Average AHT (sec)"] + [c for c in OPTIONAL_NUMERIC_COLUMNS if c in df.columns]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def normalize(series: pd.Series, lower_is_better: bool) -> pd.Series:
    """Min-max normalize a series to 0-100. Handles constant / missing values safely."""
    s = series.astype(float)
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series([70.0] * len(s), index=s.index)  # neutral score if no variation
    if lower_is_better:
        return 100 * (hi - s) / (hi - lo)
    return 100 * (s - lo) / (hi - lo)


def compute_scores(cat_df: pd.DataFrame) -> pd.DataFrame:
    """Compute a transparent 0-100 performance score for each agent row in a category."""
    d = cat_df.copy()

    has_quality = "Quality Score (%)" in d.columns and d["Quality Score (%)"].notna().any()
    has_fcr = "FCR (%)" in d.columns and d["FCR (%)"].notna().any()
    has_transfer = "Transfer Rate (%)" in d.columns and d["Transfer Rate (%)"].notna().any()

    d["_aht_score"] = normalize(d["Average AHT (sec)"], lower_is_better=True)
    d["_quality_score"] = normalize(d["Quality Score (%)"], lower_is_better=False) if has_quality else 70.0
    d["_fcr_score"] = normalize(d["FCR (%)"], lower_is_better=False) if has_fcr else 70.0
    d["_transfer_score"] = normalize(d["Transfer Rate (%)"], lower_is_better=True) if has_transfer else 70.0

    # Re-normalize weights if some components are unavailable, so total is still 0-100
    weights = {"_aht_score": W_AHT}
    if has_quality: weights["_quality_score"] = W_QUALITY
    if has_fcr: weights["_fcr_score"] = W_FCR
    if has_transfer: weights["_transfer_score"] = W_TRANSFER
    total_w = sum(weights.values())
    weights = {k: v / total_w for k, v in weights.items()}

    d["Performance Score"] = sum(d[col] * w for col, w in weights.items())
    d["Performance Score"] = d["Performance Score"].round(1)
    return d


def eligible_pool(cat_df: pd.DataFrame) -> pd.DataFrame:
    if "Availability (%)" in cat_df.columns and cat_df["Availability (%)"].notna().any():
        pool = cat_df[cat_df["Availability (%)"] >= AVAILABILITY_THRESHOLD]
        if pool.empty:  # fall back so the demo never dead-ends
            pool = cat_df
        return pool
    return cat_df


def bar(pct, color="#2563a8"):
    pct = max(0, min(100, pct))
    st.markdown(f"""
    <div style="background:#eef1f6; border-radius:6px; height:10px; width:100%; margin-bottom:2px;">
        <div style="background:{color}; width:{pct}%; height:10px; border-radius:6px;"></div>
    </div>
    """, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# SCREEN 2 — UPLOAD
# --------------------------------------------------------------------------
st.markdown('<div class="section-title">📤 Upload Agent Performance Data</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload the agent performance Excel file (.xlsx or .xls)",
    type=["xlsx", "xls"],
    label_visibility="collapsed",
)

st.caption("Demo uses synthetic agent-performance data. Production implementation can use "
           "historical operational data and an ML-based AHT prediction model.")

if uploaded_file is None:
    st.info("⬆️ Upload the AHT_Booster_Demo_Input.xlsx file (or your own agent performance "
            "file with the required columns) to begin.")
    with st.expander("Required columns"):
        st.write(REQUIRED_COLUMNS)
    st.stop()

try:
    raw_df = pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"Could not read the uploaded file. Please upload a valid Excel file. Details: {e}")
    st.stop()

ok, msg = validate_dataframe(raw_df)
if not ok:
    st.error(f"⚠️ {msg}")
    st.stop()

df = coerce_numeric(raw_df.dropna(subset=REQUIRED_COLUMNS).copy())
df = df.dropna(subset=["Average AHT (sec)"])

if df.empty:
    st.error("⚠️ No valid numeric AHT data found after cleaning. Please check the file.")
    st.stop()

n_agents = df["Agent ID"].nunique()
n_records = len(df)
categories = sorted(df["Call Category"].dropna().unique().tolist())

c1, c2, c3 = st.columns(3)
c1.markdown(f'<div class="kpi-card"><div class="kpi-label">Agents Loaded</div>'
            f'<div class="kpi-value">{n_agents}</div></div>', unsafe_allow_html=True)
c2.markdown(f'<div class="kpi-card"><div class="kpi-label">Records Loaded</div>'
            f'<div class="kpi-value">{n_records}</div></div>', unsafe_allow_html=True)
c3.markdown(f'<div class="kpi-card"><div class="kpi-label">Call Categories</div>'
            f'<div class="kpi-value">{len(categories)}</div></div>', unsafe_allow_html=True)

st.success("✅ File loaded successfully.")

# --------------------------------------------------------------------------
# SCREEN 3 — INPUTS
# --------------------------------------------------------------------------
st.markdown('<div class="section-title">⚙️ Configure Your Query</div>', unsafe_allow_html=True)

in1, in2 = st.columns([1, 1.4])
with in1:
    num_calls = st.number_input("Number of Calls", min_value=1, max_value=100000, value=8, step=1)
with in2:
    selected_category = st.selectbox("Select Call Category", categories)

# --------------------------------------------------------------------------
# SCREEN 4 — BUTTON
# --------------------------------------------------------------------------
run = st.button("🔍 FIND BEST AGENT", type="primary", use_container_width=True)

if "has_run" not in st.session_state:
    st.session_state.has_run = False
if run:
    st.session_state.has_run = True

if not st.session_state.has_run:
    st.stop()

# --------------------------------------------------------------------------
# CORE ANALYSIS (recomputed every run, directly from the uploaded Excel)
# --------------------------------------------------------------------------
cat_df = df[df["Call Category"] == selected_category].copy()

if cat_df.empty:
    st.error(f"No records found for category '{selected_category}'.")
    st.stop()

category_avg_aht = float(cat_df["Average AHT (sec)"].mean())

pool = eligible_pool(cat_df)
scored = compute_scores(pool)
scored = scored.sort_values(["Performance Score", "Average AHT (sec)"], ascending=[False, True])

best = scored.iloc[0]
best_aht = float(best["Average AHT (sec)"])

saving_per_call = category_avg_aht - best_aht
total_seconds_saved = saving_per_call * num_calls
total_minutes_saved = total_seconds_saved / 60.0

# --------------------------------------------------------------------------
# SCREEN 5 — BUSINESS IMPACT
# --------------------------------------------------------------------------
st.markdown('<div class="section-title">📊 Business Impact</div>', unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
k1.markdown(f'<div class="kpi-card"><div class="kpi-label">Number of Calls</div>'
            f'<div class="kpi-value">{num_calls:,}</div></div>', unsafe_allow_html=True)
k2.markdown(f'<div class="kpi-card"><div class="kpi-label">Category Avg AHT</div>'
            f'<div class="kpi-value">{category_avg_aht:.0f}<span style="font-size:0.9rem;"> sec</span></div></div>',
            unsafe_allow_html=True)
k3.markdown(f'<div class="kpi-card"><div class="kpi-label">Recommended Agent AHT</div>'
            f'<div class="kpi-value">{best_aht:.0f}<span style="font-size:0.9rem;"> sec</span></div></div>',
            unsafe_allow_html=True)
k4.markdown(f'<div class="kpi-card"><div class="kpi-label">Total Time Saved</div>'
            f'<div class="kpi-value highlight">{total_minutes_saved:.1f}<span style="font-size:0.9rem;"> min</span></div></div>',
            unsafe_allow_html=True)

direction = "save" if total_minutes_saved >= 0 else "cost an additional"
impact_sentence = (
    f"Routing <b>{num_calls}</b> {selected_category} calls to the recommended agent could "
    f"potentially {direction} approximately <b>{abs(total_minutes_saved):.1f} minutes</b> "
    f"compared with the category average."
)
st.markdown(f'<div class="impact-banner">💡 <b>Potential AHT Impact</b><br>{impact_sentence}</div>',
            unsafe_allow_html=True)

# --------------------------------------------------------------------------
# SCREEN 6 — BEST AGENT
# --------------------------------------------------------------------------
st.markdown('<div class="section-title">🏆 Best Fit Agent</div>', unsafe_allow_html=True)

bc1, bc2 = st.columns([1.3, 1])
with bc1:
    st.markdown(f"""
    <div class="best-agent-card">
        <div class="best-agent-title">Best Fit Agent</div>
        <div class="best-agent-name">{best['Agent Name']}</div>
        <div class="best-agent-id">Agent ID: {best['Agent ID']}</div>
        <hr style="border-color:#f2e6bd; margin: 0.8rem 0;">
        <div class="metric-row"><span>Selected Call Category</span><span>{selected_category}</span></div>
        <div class="metric-row"><span>Average AHT</span><span>{best_aht:.0f} sec</span></div>
        <div class="metric-row"><span>Performance Score</span><span>{best['Performance Score']:.1f} / 100</span></div>
        <div class="metric-row"><span>Saving Per Call</span><span>{saving_per_call:.0f} sec</span></div>
        <div class="metric-row"><span>Total Estimated Saving</span><span>{total_minutes_saved:.1f} min ({num_calls} calls)</span></div>
    </div>
    """, unsafe_allow_html=True)

with bc2:
    st.markdown("**Why this agent?**")
    aht_pct = float(best.get("_aht_score", 70))
    quality_val = best.get("Quality Score (%)", np.nan)
    fcr_val = best.get("FCR (%)", np.nan)
    transfer_val = best.get("Transfer Rate (%)", np.nan)
    avail_val = best.get("Availability (%)", np.nan)

    st.caption(f"AHT performance — {aht_pct:.0f}/100 (relative to category)")
    bar(aht_pct, "#2563a8")
    if pd.notna(quality_val):
        st.caption(f"Quality — {quality_val:.1f}%")
        bar(quality_val, "#1c8a4c")
    if pd.notna(fcr_val):
        st.caption(f"FCR — {fcr_val:.1f}%")
        bar(fcr_val, "#7c3aed")
    if pd.notna(transfer_val):
        st.caption(f"Transfer Rate — {transfer_val:.1f}% (lower is better)")
        bar(max(0, 100 - transfer_val * 3), "#e08d00")
    if pd.notna(avail_val):
        st.caption(f"Availability — {avail_val:.1f}%")
        bar(avail_val, "#0f9d8b")

# --------------------------------------------------------------------------
# SCREEN 7 — TOP 5 AGENTS
# --------------------------------------------------------------------------
st.markdown('<div class="section-title">📋 Top 5 Recommended Agents</div>', unsafe_allow_html=True)

top5 = scored.head(5).reset_index(drop=True)
top5.insert(0, "Rank", range(1, len(top5) + 1))

display_cols = ["Rank", "Agent Name", "Agent ID", "Average AHT (sec)", "Performance Score"]
for opt_col in ["Quality Score (%)", "FCR (%)", "Transfer Rate (%)", "Availability (%)"]:
    if opt_col in top5.columns:
        display_cols.append(opt_col)

st.dataframe(
    top5[display_cols].rename(columns={"Average AHT (sec)": "AHT (sec)"}),
    use_container_width=True,
    hide_index=True,
)

# --------------------------------------------------------------------------
# HOW IT WORKS
# --------------------------------------------------------------------------
with st.expander("ℹ️ How it works — recommendation logic"):
    st.markdown(f"""
This is a transparent, rule-based scoring engine (no black box):

1. **Filter to the selected call category** from the uploaded Excel data.
2. **Eligibility filter:** agents with Availability below {AVAILABILITY_THRESHOLD:.0f}% are
   excluded when availability data is present (falls back to the full pool if everyone is below
   the threshold, so the demo always returns a result).
3. **Normalize each metric to a 0–100 scale** within the category (min–max normalization):
   - Lower **AHT** → higher score
   - Higher **Quality Score** → higher score
   - Higher **FCR** → higher score
   - Lower **Transfer Rate** → higher score
4. **Weighted final score:**
   - AHT: **{W_AHT*100:.0f}%**
   - Quality: **{W_QUALITY*100:.0f}%**
   - FCR: **{W_FCR*100:.0f}%**
   - Transfer Rate: **{W_TRANSFER*100:.0f}%**
   (Weights are automatically re-balanced if a metric column is missing from the file.)
5. **Rank agents** by final score (ties broken by lower AHT) and recommend the top agent.
6. **Time savings:**
   `Saving per call = Category Average AHT − Recommended Agent AHT`
   `Total seconds saved = Saving per call × Number of Calls`
   `Total minutes saved = Total seconds saved / 60`

Every number on this page is recalculated live from the uploaded file — nothing is hardcoded.
Upload a different Excel file with the same required columns and the recommendation, ranking,
and savings will change accordingly.
    """)

st.markdown(
    '<div class="footnote">Demo uses synthetic agent-performance data. Production implementation '
    'can use historical operational data and an ML-based AHT prediction model.</div>',
    unsafe_allow_html=True,
)
