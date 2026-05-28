"""
app.py — O2C Velocity Dashboard (Streamlit)
Upload O2C_v1.xlsx to explore quote-level velocity across 16 process steps.
"""

import streamlit as st
import pandas as pd

from data_pipeline import build_velocity_table, build_standalone_html, PAIR_LABELS, STEP_LABELS

st.set_page_config(page_title="O2C Velocity Dashboard", layout="wide")

# ── File upload gate ──────────────────────────────────────────────────────────
st.title("O2C Velocity Dashboard")
uploaded = st.file_uploader("Upload O2C_v1.xlsx to begin", type=["xlsx"])
if not uploaded:
    st.stop()

with st.spinner("Processing data…"):
    df = build_velocity_table(uploaded.getvalue())

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Display")
    basis = st.radio("Time basis", ["Calendar", "Business"])
    unit  = st.radio("Unit", ["Days", "Hours"])

    if basis == "Calendar":
        src_prefix = "delta_cal_h"
        factor = 1.0 if unit == "Hours" else 1 / 24
    else:
        src_prefix = "delta_biz_d"
        factor = 8.0 if unit == "Hours" else 1.0

    unit_label = f"{basis} {unit}"

    st.divider()
    st.header("Filters")
    rework   = st.radio("Rework", ["All", "Clean only", "Reworked only"])
    nsct     = st.radio("NSCT Review", ["All", "With NSCT", "Without NSCT"])
    outcomes = st.multiselect(
        "Outcome",
        ["Accepted", "Rejected", "Denied", "In Progress"],
        default=[],
    )
    hide_multi = st.checkbox("Exclude multi-order from averages", value=True)

    st.divider()


# ── Filter logic ──────────────────────────────────────────────────────────────
def apply_filters(
    df: pd.DataFrame,
    rework: str,
    nsct: str,
    outcomes: list,
    hide_multi: bool,
) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    if rework == "Clean only":
        mask &= ~df["rework_flag"]
    elif rework == "Reworked only":
        mask &= df["rework_flag"]
    if nsct == "With NSCT":
        mask &= df["nsct_flag"]
    elif nsct == "Without NSCT":
        mask &= ~df["nsct_flag"]
    if outcomes:
        mask &= df["outcome"].isin(outcomes)

    filt = df[mask].copy()

    if hide_multi:
        for i in [13, 14, 15]:
            filt.loc[filt["multi_order_flag"], f"delta_cal_h_{i}"] = float("nan")
            filt.loc[filt["multi_order_flag"], f"delta_biz_d_{i}"] = float("nan")
    return filt


filt = apply_filters(df, rework, nsct, outcomes, hide_multi)

with st.sidebar:
    st.caption(f"**{len(filt):,}** / {len(df):,} quotes matched")


# ── View A builder ────────────────────────────────────────────────────────────
def build_view_a_rows(
    filt: pd.DataFrame,
    src_prefix: str,
    factor: float,
    unit_label: str,
    rework: str,
) -> list[dict]:
    rows = []
    for i, label in enumerate(PAIR_LABELS, start=1):
        raw_col = f"{src_prefix}_{i}"
        vals = pd.to_numeric(filt[raw_col], errors="coerce").dropna() * factor
        step_label = f"⚠️ {label}" if i == 15 else label

        row = {
            "Step Pair": step_label,
            f"Avg ({unit_label})":    round(float(vals.mean()),   1) if len(vals) else None,
            f"Median ({unit_label})": round(float(vals.median()), 1) if len(vals) else None,
            "n": int(len(vals)),
        }

        if rework == "All":
            clean_raw  = pd.to_numeric(
                filt.loc[~filt["rework_flag"], raw_col], errors="coerce"
            ).dropna() * factor
            rework_raw = pd.to_numeric(
                filt.loc[ filt["rework_flag"], raw_col], errors="coerce"
            ).dropna() * factor
            row["Avg (Clean)"]  = round(float(clean_raw.mean()),  1) if len(clean_raw)  else None
            row["Avg (Rework)"] = round(float(rework_raw.mean()), 1) if len(rework_raw) else None

        rows.append(row)
    return rows


# ── Quote card renderer ───────────────────────────────────────────────────────
_DOCUSIGN_STEPS = {10, 11, 12}   # 1-indexed steps sourced from DocuSign


def render_quote_card(
    row: pd.Series,
    src_prefix: str,
    factor: float,
    unit_label: str,
) -> None:
    qnum    = row.get("quote_number")
    opp     = row.get("opportunity_id") or "—"
    outcome = row.get("outcome", "—")
    rw_flag = bool(row.get("rework_flag", False))
    rw_lbl  = f"Yes — {row.get('rework_stage', '')}" if rw_flag else "No"
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
                pair_n = i - 1
                raw_val = row.get(f"{src_prefix}_{pair_n}")
                try:
                    v = float(raw_val)
                    if pd.isna(v):
                        raise ValueError
                    disp = v * factor
                    delta_str = f"{disp:.1f}"
                except (TypeError, ValueError):
                    delta_str = (
                        "Pending"
                        if (i in _DOCUSIGN_STEPS and has_contract)
                        else "—"
                    )

            timeline.append({
                "#": i,
                "Event": display_label,
                "Timestamp": ts_str,
                f"Δ {unit_label}": delta_str,
            })

        st.table(pd.DataFrame(timeline).set_index("#"))


# ── Build View A data ─────────────────────────────────────────────────────────
view_a_rows = build_view_a_rows(filt, src_prefix, factor, unit_label, rework)
view_a_df   = pd.DataFrame(view_a_rows)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_a, tab_b = st.tabs(["📊 Stage Velocity", "🔍 Quote Timeline"])

with tab_a:
    if rework == "All" and "Avg (Rework)" in view_a_df.columns:
        def _highlight(row):
            styles = [""] * len(row)
            c = row.get("Avg (Clean)")
            r = row.get("Avg (Rework)")
            if pd.notna(c) and pd.notna(r) and float(r) > float(c):
                idx = list(row.index).index("Avg (Rework)")
                styles[idx] = (
                    "background-color:#ffe0e0;color:#c0392b;font-weight:bold"
                )
            return styles

        st.dataframe(
            view_a_df.style.apply(_highlight, axis=1),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.dataframe(view_a_df, use_container_width=True, hide_index=True)

    st.caption(
        "⚠️ Step 15→16 (Deployment Closed) is manually set "
        "and may lag actual install date."
    )

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
            st.warning("Enter a numeric quote number (digits only, no letters other than leading Q).")
    elif opp_input:
        matches = df[df["opportunity_id"] == opp_input.strip()]

    if not matches.empty:
        for _, row in matches.iterrows():
            render_quote_card(row, src_prefix, factor, unit_label)
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
