"""
app.py — O2C Velocity Dashboard (Streamlit)

Upload O2C_v1.xlsx to explore quote-level velocity across 16 process steps.
The on-screen content mirrors the downloadable HTML report:
  • Sidebar filters: Time Basis, Unit, Object, Month, Rework, NSCT, Outcome
  • View A (Aggregate Velocity): step-pair table, bar chart, time-distribution
    heatmap, and an end-to-end phase summary
  • View B (Per-Quote Timeline): search by quote number or opportunity ID
"""

import pandas as pd
import altair as alt
import streamlit as st

from data_pipeline import (
    build_velocity_table,
    build_standalone_html,
    PAIR_LABELS,
    STEP_LABELS,
    PHASE_LABELS,
)

st.set_page_config(page_title="O2C Velocity Dashboard", layout="wide")

# ── Object → step-pair mapping (matches the HTML OBJECT_PAIRS) ─────────────────
OBJECT_PAIRS = {
    "All":      list(range(15)),
    "Quote":    [0, 1, 2, 3, 4, 5, 6, 7],
    "DocuSign": [8, 9, 10],
    "Contract": [11],
    "Order":    [12, 13, 14],
}

# Object that owns each of the 15 step pairs (for chart colour-coding)
_OBJECT_RANGE = {
    "Quote":    "#3b82f6",
    "DocuSign": "#7c3aed",
    "Contract": "#0891b2",
    "Order":    "#059669",
}


def pair_object(i: int) -> str:
    if i < 8:
        return "Quote"
    if i < 11:
        return "DocuSign"
    if i == 11:
        return "Contract"
    return "Order"


def month_label(ym: str) -> str:
    if ym == "All":
        return "All"
    y, m = ym.split("-")
    names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
             "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return f"{names[int(m) - 1]} {y}"


def fmt(v) -> str:
    return "—" if v is None or pd.isna(v) else f"{v:.1f}"


# ── File upload gate ──────────────────────────────────────────────────────────
st.title("Order-to-Cash Velocity Dashboard")
uploaded = st.file_uploader("Upload O2C_v1.xlsx to begin", type=["xlsx"])
if not uploaded:
    st.stop()

with st.spinner("Processing data…"):
    df = build_velocity_table(uploaded.getvalue())

# ── Sidebar filters ─────────────────────────────────────────────────────────--
with st.sidebar:
    st.header("Display")
    basis = st.radio("Time Basis", ["Calendar", "Business"], horizontal=True)
    unit  = st.radio("Unit", ["Days", "Hours"], horizontal=True)

    if basis == "Calendar":
        src_prefix, phase_prefix = "delta_cal_h", "phase_cal"
        factor = 1.0 if unit == "Hours" else 1 / 24
    else:
        src_prefix, phase_prefix = "delta_biz_d", "phase_biz"
        factor = 8.0 if unit == "Hours" else 1.0

    unit_label  = f"{basis} {unit}"
    unit_suffix = "h" if unit == "Hours" else "d"

    st.divider()
    st.header("Filters")

    obj = st.radio("Object", list(OBJECT_PAIRS.keys()))

    month_vals = sorted(df["s01"].dropna().dt.strftime("%Y-%m").unique().tolist())
    month = st.selectbox("Month", ["All"] + month_vals, format_func=month_label)

    rework   = st.radio("Rework", ["All", "Reworked only", "Clean only"])
    nsct     = st.radio("NSCT Review", ["All", "With NSCT", "Without NSCT"])
    outcomes = st.multiselect(
        "Outcome",
        ["Accepted", "Rejected", "Denied", "In Progress"],
        default=[],
    )


# ── Population filter (drives all averages) ────────────────────────────────────
def apply_population_filters(
    df: pd.DataFrame,
    month: str,
    rework: str,
    nsct: str,
    outcomes: list,
) -> pd.DataFrame:
    """Scope the quote population. Multi-order exclusion (pairs 13–15) is already
    baked into the pipeline, so no toggle is needed here."""
    mask = pd.Series(True, index=df.index)
    if month != "All":
        mask &= df["s01"].dt.strftime("%Y-%m") == month
    if rework == "Reworked only":
        mask &= df["rework_flag"]
    elif rework == "Clean only":
        mask &= ~df["rework_flag"]
    if nsct == "With NSCT":
        mask &= df["nsct_flag"]
    elif nsct == "Without NSCT":
        mask &= ~df["nsct_flag"]
    if outcomes:
        mask &= df["outcome"].isin(outcomes)
    return df[mask].copy()


pop = apply_population_filters(df, month, rework, nsct, outcomes)

with st.sidebar:
    st.caption(f"**{len(pop):,}** / {len(df):,} quotes matched")


# ── Aggregation helpers ────────────────────────────────────────────────────────
def pair_avg(pop: pd.DataFrame, prefix: str, i: int, factor: float):
    vals = pd.to_numeric(pop[f"{prefix}_{i + 1}"], errors="coerce").dropna() * factor
    if not len(vals):
        return None, None, 0
    return float(vals.mean()), float(vals.median()), int(len(vals))


def phase_avg(pop: pd.DataFrame, prefix: str, k: int, factor: float):
    vals = pd.to_numeric(pop[f"{prefix}_{k}"], errors="coerce").dropna() * factor
    if not len(vals):
        return None, None, 0
    return float(vals.mean()), float(vals.median()), int(len(vals))


# Compute all 15 pair rows for the active population
all_rows = []
for i in range(15):
    a, md, n = pair_avg(pop, src_prefix, i, factor)
    all_rows.append({"pair_idx": i, "label": PAIR_LABELS[i], "avg": a, "median": md, "n": n})

# When "Reworked only": compute the clean-path comparison population
clean_rows = None
if rework == "Reworked only":
    clean_pop = apply_population_filters(df, month, "Clean only", nsct, outcomes)
    clean_rows = [pair_avg(clean_pop, src_prefix, i, factor)[0] for i in range(15)]


# ── View builders ──────────────────────────────────────────────────────────────
def render_step_pair_table(vis_idx: list):
    """Step-pair table scoped to the object filter's visible rows."""
    table = []
    for i in vis_idx:
        r = all_rows[i]
        warn = " ⚠️" if i == 14 else ""
        row = {
            "Step Pair":               f"{r['label']}{warn}",
            f"Avg ({unit_label})":     fmt(r["avg"]),
            f"Median ({unit_label})":  fmt(r["median"]),
            "n":                       r["n"],
        }
        if clean_rows is not None:
            ca = clean_rows[i]
            delta = (r["avg"] - ca) if (r["avg"] is not None and ca is not None) else None
            row["vs. Clean Path"] = (
                "—" if delta is None
                else f"+{delta:.1f}{unit_suffix}" if delta > 0
                else f"{delta:.1f}{unit_suffix}"
            )
        table.append(row)
    st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)


def render_rework_callout():
    """List step pairs where reworked quotes are slower than the clean path."""
    slower = []
    for i in range(15):
        r, ca = all_rows[i], clean_rows[i]
        if r["avg"] is not None and ca is not None and r["avg"] > ca:
            slower.append({
                "Step Pair":  r["label"],
                "Rework Avg": f"{r['avg']:.1f}{unit_suffix}",
                "Clean Avg":  f"{ca:.1f}{unit_suffix}",
                "Delta":      f"+{r['avg'] - ca:.1f}{unit_suffix}",
            })
    if slower:
        st.warning("**Rework Cost — steps where reworked quotes are slower than clean-path quotes**")
        st.dataframe(pd.DataFrame(slower), use_container_width=True, hide_index=True)
    else:
        st.info("No step pairs where reworked quotes average more time than clean-path quotes.")


def render_bar_chart():
    """Avg time per step pair — all 15 pairs, object colour-coded."""
    order_list = []
    data = []
    for i in range(15):
        r = all_rows[i]
        lbl = f"{r['label']} ⚠️" if i == 14 else r["label"]
        order_list.append(lbl)
        data.append({
            "Step Pair": lbl,
            "Avg":       r["avg"] if r["avg"] is not None else 0.0,
            "AvgLabel":  fmt(r["avg"]) + (unit_suffix if r["avg"] is not None else ""),
            "Object":    pair_object(i),
            "n":         r["n"],
        })
    cdf = pd.DataFrame(data)
    bars = alt.Chart(cdf).mark_bar(opacity=0.85, cornerRadius=3).encode(
        y=alt.Y("Step Pair:N", sort=order_list, title=None,
                axis=alt.Axis(labelLimit=320, labelFontSize=11)),
        x=alt.X("Avg:Q", title=f"Avg ({unit_label})"),
        color=alt.Color(
            "Object:N",
            scale=alt.Scale(domain=list(_OBJECT_RANGE.keys()),
                            range=list(_OBJECT_RANGE.values())),
            legend=alt.Legend(orient="top", title=None),
        ),
        tooltip=[alt.Tooltip("Step Pair:N"),
                 alt.Tooltip("Avg:Q", format=".1f", title=f"Avg ({unit_label})"),
                 alt.Tooltip("n:Q")],
    )
    text = alt.Chart(cdf).mark_text(align="left", dx=3, fontSize=11, fontWeight="bold").encode(
        y=alt.Y("Step Pair:N", sort=order_list),
        x=alt.X("Avg:Q"),
        text=alt.Text("AvgLabel:N"),
    )
    st.altair_chart((bars + text).properties(height=470), use_container_width=True)


def render_heatmap():
    """Proportional time-distribution bar — segment width ∝ avg time, colour = heat.
    Step name + value appear on hover (tooltip)."""
    avgs = [r["avg"] if r["avg"] is not None else 0.0 for r in all_rows]
    total = sum(avgs) or 1.0
    data, cum = [], 0.0
    for i in range(15):
        a = avgs[i]
        start, cum = cum, cum + a
        lbl = f"{all_rows[i]['label']} ⚠️" if i == 14 else all_rows[i]["label"]
        data.append({
            "Step Pair": lbl,
            "Step":      i + 1,
            "Avg":       all_rows[i]["avg"],
            "AvgDisp":   all_rows[i]["avg"] if all_rows[i]["avg"] is not None else 0.0,
            "Pct":       round(a / total * 100, 1),
            "start":     start,
            "end":       cum,
            "mid":       (start + cum) / 2,
        })
    hdf = pd.DataFrame(data)
    base = alt.Chart(hdf)
    bars = base.mark_bar(stroke="white", strokeWidth=1.5).encode(
        x=alt.X("start:Q", title=None, axis=None, scale=alt.Scale(domain=[0, total])),
        x2="end:Q",
        color=alt.Color(
            "AvgDisp:Q",
            scale=alt.Scale(range=["#bfdbfe", "#fde047", "#dc2626"]),
            legend=alt.Legend(orient="top", title=f"Avg ({unit_label})"),
        ),
        tooltip=[alt.Tooltip("Step Pair:N", title="Step"),
                 alt.Tooltip("Avg:Q", format=".1f", title=f"Avg ({unit_label})"),
                 alt.Tooltip("Pct:Q", format=".1f", title="% of cycle")],
    )
    labels = base.mark_text(fontSize=11, fontWeight="bold", color="#1e293b").encode(
        x=alt.X("mid:Q", scale=alt.Scale(domain=[0, total])),
        text=alt.Text("Step:N"),
    )
    st.altair_chart((bars + labels).properties(height=90), use_container_width=True)
    st.caption("Each segment's width is its share of the average cycle. Hover for step name, value and %.")


def render_summary():
    """End-to-end phase summary (mirrors the HTML)."""
    pstats = [phase_avg(pop, phase_prefix, k, factor) for k in range(4)]  # phases 0–3

    def sum_avg(idxs):
        parts = [pstats[k][0] for k in idxs]
        return None if any(p is None for p in parts) else sum(parts)

    rows = [
        {"Phase": PHASE_LABELS[0], "Avg": fmt(pstats[0][0]), "Median": fmt(pstats[0][1]), "n": pstats[0][2]},
        {"Phase": PHASE_LABELS[1], "Avg": fmt(pstats[1][0]), "Median": fmt(pstats[1][1]), "n": pstats[1][2]},
        {"Phase": PHASE_LABELS[2], "Avg": fmt(pstats[2][0]), "Median": fmt(pstats[2][1]), "n": pstats[2][2]},
        {"Phase": "Created → Contract Activated (subtotal)",
         "Avg": fmt(sum_avg([0, 1, 2])), "Median": "—", "n": "—"},
        {"Phase": PHASE_LABELS[3], "Avg": fmt(pstats[3][0]), "Median": fmt(pstats[3][1]), "n": pstats[3][2]},
        {"Phase": "Full Cycle — Created → Deployed",
         "Avg": fmt(sum_avg([0, 1, 2, 3])), "Median": "—", "n": "—"},
    ]
    sub_idx, full_idx = 3, 5

    def _style(row):
        if row.name in (sub_idx, full_idx):
            return ["font-weight:700;background-color:#f1f5f9"] * len(row)
        return [""] * len(row)

    styled = pd.DataFrame(rows).style.apply(_style, axis=1)
    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.caption(
        "Subtotal and Full Cycle are **sums of the phase averages** (not the average of "
        "individual end-to-end durations), so every phase contributes regardless of how "
        "far each quote progressed."
    )


# ── Quote card (View B) ─────────────────────────────────────────────────────────
_DOCUSIGN_STEPS = {10, 11, 12}


def render_quote_card(row: pd.Series):
    qnum     = row.get("quote_number")
    opp      = row.get("opportunity_id") or "—"
    outcome  = row.get("outcome", "—")
    rw_flag  = bool(row.get("rework_flag", False))
    rw_lbl   = f"Yes — {row.get('rework_stage', '')}" if rw_flag else "No"
    nsct_lbl = "Yes" if row.get("nsct_flag") else "No"
    mo_lbl   = "Yes" if row.get("multi_order_flag") else "No"

    header = (
        f"Q{qnum}  ·  {outcome}  ·  Rework: {rw_lbl}"
        f"  ·  NSCT: {nsct_lbl}  ·  Multi-Order: {mo_lbl}"
    )
    with st.expander(header, expanded=True):
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Quote",       f"Q{qnum}")
        c2.metric("Opportunity", str(opp)[:24] if opp and opp != "—" else "—")
        c3.metric("Outcome",     outcome)
        c4.metric("Rework",      rw_lbl)
        c5.metric("NSCT / Multi-Order", f"{nsct_lbl} / {mo_lbl}")

        has_contract = pd.notna(row.get("contract_number"))
        timeline = []
        for i, label in enumerate(STEP_LABELS, start=1):
            ts_raw = row.get(f"s{i:02d}")
            if pd.notna(ts_raw):
                ts_str = pd.Timestamp(ts_raw).strftime("%Y-%m-%d %H:%M")
            elif i in _DOCUSIGN_STEPS and has_contract:
                ts_str = "Pending"
            else:
                ts_str = "—"

            display_label = f"⚠️ {label}" if i == 16 else label

            if i == 1:
                delta_str = "—"
            else:
                raw_val = row.get(f"{src_prefix}_{i - 1}")
                try:
                    v = float(raw_val)
                    if pd.isna(v):
                        raise ValueError
                    delta_str = f"{v * factor:.1f}"
                except (TypeError, ValueError):
                    delta_str = "Pending" if (i in _DOCUSIGN_STEPS and has_contract) else "—"

            timeline.append({
                "#": i,
                "Event": display_label,
                "Timestamp": ts_str,
                f"Δ {unit_label}": delta_str,
            })
        st.table(pd.DataFrame(timeline).set_index("#"))


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_a, tab_b = st.tabs(["📊 Aggregate Velocity", "🔍 Per-Quote Timeline"])

with tab_a:
    if rework == "Reworked only":
        render_rework_callout()

    st.subheader("Step-Pair Velocity")
    if obj != "All":
        st.caption(f"Table scoped to **{obj}** steps. Charts and summary always reflect the full pipeline.")
    render_step_pair_table(OBJECT_PAIRS[obj])
    st.caption("⚠️ Step 15→16 (Deployment Closed) is manually set and may lag actual install date.")

    st.subheader(f"Avg Time per Step Pair — {unit_label}")
    render_bar_chart()

    st.subheader("Time Distribution — % of Avg Cycle")
    render_heatmap()

    st.subheader(f"End-to-End Summary — {unit_label}")
    render_summary()

with tab_b:
    col1, col2 = st.columns(2)
    qnum_input = col1.text_input("Quote Number (e.g. 57341 or Q57341)")
    opp_input  = col2.text_input("Opportunity ID")

    matches = pd.DataFrame()
    if qnum_input:
        q_clean = qnum_input.strip().lstrip("Qq")
        try:
            matches = df[df["quote_number"] == int(q_clean)]
        except ValueError:
            st.warning("Enter a numeric quote number (digits only, optional leading Q).")
    elif opp_input:
        matches = df[df["opportunity_id"] == opp_input.strip()]

    if not matches.empty:
        for _, row in matches.iterrows():
            render_quote_card(row)
    elif qnum_input or opp_input:
        st.info("No matching quotes found.")

# ── Download ──────────────────────────────────────────────────────────────────
st.divider()
html_snap = build_standalone_html(df)
st.download_button(
    "⬇️ Download HTML Report",
    data=html_snap,
    file_name="o2c_velocity_dashboard.html",
    mime="text/html",
)
