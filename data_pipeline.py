"""
data_pipeline.py — O2C Velocity Analysis
Loads O2C_v1.xlsx (uploaded as bytes), joins all tabs, extracts 16 event
timestamps per quote, computes segmentation flags and step-pair deltas.

Ground truth: resources/velocity-analysis-reference.md §10 (column names,
join keys, step filter logic).  §2 / §3 of that doc describe the original
anonymized files and are NOT used here.
"""

import io
import json
import re
import math
import warnings

import numpy as np
import pandas as pd
import holidays

try:
    import streamlit as st
    _cache = st.cache_data
except ImportError:
    def _cache(fn):          # allow import outside Streamlit for unit tests
        return fn

# ── Step / pair labels ────────────────────────────────────────────────────────

STEP_LABELS = [
    "Quote Created",        # s01
    "Tech Review",          # s02
    "Tech Approved",        # s03
    "Commercial Review",    # s04
    "Commercial Approved",  # s05
    "NSCT Review",          # s06
    "Fully Approved",       # s07
    "Presented",            # s08
    "Accepted",             # s09
    "Signature Sent",       # s10  (sourced from the MCF signature status)
    "Customer Signed",      # s11
    "Counter Signature",    # s12  (was "Fully Executed")
    "Contract Activated",   # s13
    "Order Activated",      # s14
    "Awaiting Install",     # s15
    "Deployment Closed",    # s16
]

PAIR_LABELS = [
    f"{STEP_LABELS[i]} to {STEP_LABELS[i + 1]}" for i in range(15)
]

# ── Phase spans (end-to-end summary) ──────────────────────────────────────────
# (start_step_idx, end_step_idx) — 0-based into the 16 steps.
_PHASE_EP = [(0, 8), (8, 11), (11, 12), (12, 15), (0, 15)]
PHASE_LABELS = [
    "Quote Phase (Created to Accepted)",
    "DocuSign Phase (Accepted to Counter Signature)",
    "Contract Phase (Counter Signature to Contract Activated)",
    "Order Phase (Contract to Deployed)",
    "Full Cycle (Created to Deployed)",
]

# ── Signature-status sourcing / resend detection ──────────────────────────────
_MCF_FIELD = "MCF_Signature_Status__c"
_MCF_SENT_VALUE = "Sent for signature"
RESEND_GAP_DAYS = 1  # DocuSign send lagging the MCF status by more than this = resend-suspected

# ── US holiday dates for business-day calculation ─────────────────────────────

_us_hols = holidays.US(years=range(2024, 2030))
US_HOLIDAY_DATES = np.array(
    [np.datetime64(str(d), "D") for d in sorted(_us_hols.keys())]
)

# ── Tab loaders ───────────────────────────────────────────────────────────────

def _load_quote_history(xls: pd.ExcelFile) -> pd.DataFrame:
    df = xls.parse("Quote History")
    rename = {
        "Quote ID":               "quote_id",
        "Quote : Quote Number":   "quote_number",
        "Quote : Quote Name":     "quote_name",
        "Quote : Opportunity ID": "opportunity_id",
        "Quote : Quote Type":     "quote_type",
        "Changed Field":          "field",
        "New Value":              "new_value",
        "Created Date":           "created_date",
    }
    df = df.rename(columns=rename)[[c for c in rename.values() if c in df.rename(columns=rename).columns]]
    df["created_date"] = pd.to_datetime(df["created_date"], utc=True).dt.tz_localize(None)
    df["quote_id"] = df["quote_id"].astype(str).str.strip()
    # new_value can be mixed types — normalise to str or NaN
    df["new_value"] = df["new_value"].where(df["new_value"].notna()).astype(str)
    df.loc[df["new_value"] == "nan", "new_value"] = None
    return df


def _load_order_history(xls: pd.ExcelFile) -> pd.DataFrame:
    df = xls.parse("Order History")
    rename = {
        "Order : Quote ID":     "quote_id",
        "Order : Order Number": "order_number",
        "Changed Field":        "field",
        "New Value":            "new_value",
        "Created Date":         "created_date",
    }
    df = df.rename(columns=rename)[[c for c in rename.values() if c in df.rename(columns=rename).columns]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df["created_date"] = pd.to_datetime(df["created_date"], format="mixed")
    df["quote_id"] = df["quote_id"].astype(str).str.strip()
    df["new_value"] = df["new_value"].where(df["new_value"].notna()).astype(str)
    df.loc[df["new_value"] == "nan", "new_value"] = None
    return df


def _load_contract_history(xls: pd.ExcelFile) -> pd.DataFrame:
    df = xls.parse("Contract History")
    rename = {
        "Contract : Quote ID":      "quote_id",
        "Contract : Contract Name": "contract_name",
        "Changed Field":            "field",
        "New Value":                "new_value",
        "Created Date":             "created_date",
    }
    df = df.rename(columns=rename)[[c for c in rename.values() if c in df.rename(columns=rename).columns]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df["created_date"] = pd.to_datetime(df["created_date"], format="mixed")
    df["quote_id"] = df["quote_id"].astype(str).str.strip()
    df["new_value"] = df["new_value"].where(df["new_value"].notna()).astype(str)
    df.loc[df["new_value"] == "nan", "new_value"] = None

    # Parse numeric contract ID from name pattern "… - C{NNNNN}"
    # The actual separator is " - C" (space-dash-space), so allow optional whitespace
    df["contract_number"] = (
        df["contract_name"]
        .str.extract(r"-\s*C(\d+)\s*$", expand=False)
        .astype("Int64")
    )
    non_parseable = df["contract_name"].notna() & df["contract_number"].isna()
    if non_parseable.any():
        bad = df.loc[non_parseable, "contract_name"].dropna().tolist()
        warnings.warn(
            f"Could not parse contract number from {len(bad)} row(s) in Contract History: {bad[:5]}"
        )
    return df


def _load_docusign(xls: pd.ExcelFile) -> pd.DataFrame:
    df = xls.parse("Docusign Status Report")
    rename = {
        "Contract Number":        "contract_number",
        "DocuSign Routing Order": "routing_order",
        "Date Sent":              "date_sent",
        "Date Signed":            "date_signed",
    }
    df = df.rename(columns=rename)[[c for c in rename.values() if c in df.rename(columns=rename).columns]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df["date_sent"]   = pd.to_datetime(df["date_sent"],   format="mixed", errors="coerce")
        df["date_signed"] = pd.to_datetime(df["date_signed"], format="mixed", errors="coerce")
    df["contract_number"] = pd.to_numeric(df["contract_number"], errors="coerce").astype("Int64")
    return df


# ── Event extraction ──────────────────────────────────────────────────────────

def _extract_quote_events(qh: pd.DataFrame) -> pd.DataFrame:
    """Steps 1–9 from Quote History. One row per quote_id."""
    s01 = (
        qh[qh["field"] == "created"]
        .groupby("quote_id", sort=False)["created_date"].min()
        .rename("s01")
    )

    status = qh[qh["field"] == "Status"]
    step_values = [
        ("s02", "TechReview"),
        ("s03", "TechApproved"),
        ("s04", "CommercialReview"),
        ("s05", "Commercial approved"),  # exact case from data
        ("s06", "NSCT Review"),
        ("s07", "Fully Approved"),
        ("s08", "Presented"),
        ("s09", "Accepted"),
    ]

    qt = s01.reset_index()
    for col, val in step_values:
        first = (
            status[status["new_value"] == val]
            .groupby("quote_id", sort=False)["created_date"].min()
            .rename(col)
        )
        qt = qt.merge(first.reset_index(), on="quote_id", how="left")

    # Opportunity ID — first non-null per quote from Quote History
    opp_map = (
        qh[qh["opportunity_id"].notna()]
        .groupby("quote_id", sort=False)["opportunity_id"].first()
        .rename("opportunity_id")
    )
    qt = qt.merge(opp_map.reset_index(), on="quote_id", how="left")

    # Quote metadata (number, name, type)
    meta = (
        qh.groupby("quote_id", sort=False)[["quote_number", "quote_name", "quote_type"]]
        .first()
        .reset_index()
    )
    qt = qt.merge(meta, on="quote_id", how="left")

    # s10 "Signature Sent" — sourced from the Quote's MCF signature status
    # ("Sent for signature"), not DocuSign Date Sent. This captures the original
    # send intent and is immune to the DocuSign void/resend survivorship bias.
    s10 = (
        qh[(qh["field"] == _MCF_FIELD) & (qh["new_value"] == _MCF_SENT_VALUE)]
        .groupby("quote_id", sort=False)["created_date"].min()
        .rename("s10")
    )
    qt = qt.merge(s10.reset_index(), on="quote_id", how="left")
    return qt


def _join_order_history(qt: pd.DataFrame, oh: pd.DataFrame) -> pd.DataFrame:
    """Steps 14–16 and order_count from Order History."""
    # s14: orderActivated event flag (no New Value filter)
    s14 = (
        oh[oh["field"] == "orderActivated"]
        .groupby("quote_id", sort=False)["created_date"].min()
        .rename("s14")
    )
    # s15: Awaiting Install
    s15 = (
        oh[
            (oh["field"] == "MCF_Deployment_status__c")
            & (oh["new_value"] == "Awaiting Install")
        ]
        .groupby("quote_id", sort=False)["created_date"].min()
        .rename("s15")
    )
    # s16: Deployment Closed
    s16 = (
        oh[
            (oh["field"] == "MCF_Deployment_status__c")
            & (oh["new_value"] == "Closed")
        ]
        .groupby("quote_id", sort=False)["created_date"].min()
        .rename("s16")
    )
    order_count = (
        oh.groupby("quote_id", sort=False)["order_number"].nunique()
        .rename("order_count")
    )

    for s in [s14, s15, s16, order_count]:
        qt = qt.merge(s.reset_index(), on="quote_id", how="left")
    qt["order_count"] = qt["order_count"].fillna(0).astype(int)
    return qt


def _join_contract_history(qt: pd.DataFrame, ch: pd.DataFrame) -> pd.DataFrame:
    """Step 13 and contract_number from Contract History."""
    s13 = (
        ch[(ch["field"] == "Status") & (ch["new_value"] == "Activated")]
        .groupby("quote_id", sort=False)["created_date"].min()
        .rename("s13")
    )
    # First parseable contract number per quote
    contract_map = (
        ch[ch["contract_number"].notna()]
        .groupby("quote_id", sort=False)["contract_number"].first()
        .rename("contract_number")
    )
    qt = qt.merge(s13.reset_index(), on="quote_id", how="left")
    qt = qt.merge(contract_map.reset_index(), on="quote_id", how="left")
    return qt


def _join_docusign(qt: pd.DataFrame, ds: pd.DataFrame) -> pd.DataFrame:
    """Steps 11–12 from DocuSign, joined on contract_number.

    s10 ("Signature Sent") is NOT taken from DocuSign — it comes from the MCF
    signature status (see _extract_quote_events). DocuSign routing-order-1
    Date Sent is kept as ``docusign_sent_1`` purely to detect voided-and-resent
    envelopes against the MCF s10 (see _compute_resend_flag).
    """
    # Ensure join key types match
    qt["contract_number"] = qt["contract_number"].astype("Int64")

    docusign_sent_1 = (
        ds[ds["routing_order"] == 1]
        .groupby("contract_number", sort=False)["date_sent"].min()
        .rename("docusign_sent_1")
    )
    s11 = (
        ds[ds["routing_order"] == 1]
        .groupby("contract_number", sort=False)["date_signed"].min()
        .rename("s11")
    )
    s12 = (
        ds[ds["routing_order"] == 2]
        .groupby("contract_number", sort=False)["date_signed"].min()
        .rename("s12")
    )
    for s in [docusign_sent_1, s11, s12]:
        qt = qt.merge(s.reset_index(), on="contract_number", how="left")

    # Reorder step columns into canonical order
    step_cols = [f"s{i:02d}" for i in range(1, 17)]
    other_cols = [c for c in qt.columns if c not in step_cols]
    qt = qt[other_cols + step_cols]
    return qt


# ── Segmentation flags ────────────────────────────────────────────────────────

def _compute_flags(qt: pd.DataFrame, qh: pd.DataFrame) -> pd.DataFrame:
    status = qh[qh["field"] == "Status"]

    # Rework: TechReview or CommercialReview appears more than once
    tech_count = (
        status[status["new_value"] == "TechReview"]
        .groupby("quote_id").size()
        .rename("tech_count")
    )
    comm_count = (
        status[status["new_value"] == "CommercialReview"]
        .groupby("quote_id").size()
        .rename("comm_count")
    )
    qt = qt.merge(tech_count.reset_index(), on="quote_id", how="left")
    qt = qt.merge(comm_count.reset_index(), on="quote_id", how="left")
    qt["tech_count"] = qt["tech_count"].fillna(0).astype(int)
    qt["comm_count"] = qt["comm_count"].fillna(0).astype(int)

    qt["rework_flag"] = (qt["tech_count"] > 1) | (qt["comm_count"] > 1)

    qt["rework_stage"] = None
    qt.loc[(qt["tech_count"] > 1) & (qt["comm_count"] <= 1), "rework_stage"] = "TechReview"
    qt.loc[(qt["tech_count"] <= 1) & (qt["comm_count"] > 1), "rework_stage"] = "CommercialReview"
    qt.loc[(qt["tech_count"] > 1)  & (qt["comm_count"] > 1),  "rework_stage"] = "Both"

    qt["rework_count"] = (
        (qt["tech_count"] - 1).clip(lower=0)
        + (qt["comm_count"] - 1).clip(lower=0)
    )

    # NSCT
    nsct_ids = set(status[status["new_value"] == "NSCT Review"]["quote_id"].unique())
    qt["nsct_flag"] = qt["quote_id"].isin(nsct_ids)

    # Outcome: last Status value, mapped
    OUTCOME_MAP = {
        "Accepted": "Accepted",
        "Rejected": "Rejected",
        "Denied":   "Denied",
    }
    last_status = (
        status.sort_values("created_date")
        .groupby("quote_id", sort=False)["new_value"].last()
        .rename("_last_status")
    )
    qt = qt.merge(last_status.reset_index(), on="quote_id", how="left")
    qt["outcome"] = qt["_last_status"].map(OUTCOME_MAP).fillna("In Progress")
    qt = qt.drop(columns=["_last_status", "tech_count", "comm_count"])

    # Multi-order
    qt["multi_order_flag"] = qt["order_count"] > 1

    return qt


# ── Delta computation ─────────────────────────────────────────────────────────

def _biz_deltas(starts: pd.Series, ends: pd.Series) -> np.ndarray:
    """Business-day count between two datetime series. NaN where either is NaT."""
    valid = starts.notna() & ends.notna()
    result = np.full(len(starts), np.nan, dtype=float)
    if valid.any():
        s = starts[valid].values.astype("datetime64[D]")
        e = ends[valid].values.astype("datetime64[D]")
        result[valid.values] = np.busday_count(s, e, holidays=US_HOLIDAY_DATES).astype(float)
    return result


def _compute_deltas(qt: pd.DataFrame) -> pd.DataFrame:
    step_cols = [f"s{i:02d}" for i in range(1, 17)]
    for n in range(1, 16):
        start = qt[step_cols[n - 1]]
        end   = qt[step_cols[n]]
        # Calendar: store as hours (float) — app divides by 24 for days
        qt[f"delta_cal_h_{n}"] = (end - start).dt.total_seconds() / 3600
        # Business: store as days (float, NaN for null)
        qt[f"delta_biz_d_{n}"] = _biz_deltas(start, end)

    # Multi-order quotes: null out pairs 13→14, 14→15, 15→16
    mo = qt["multi_order_flag"]
    for n in [13, 14, 15]:
        qt.loc[mo, f"delta_cal_h_{n}"] = np.nan
        qt.loc[mo, f"delta_biz_d_{n}"] = np.nan

    return qt


def _compute_resend_flag(qt: pd.DataFrame) -> pd.DataFrame:
    """Flag quotes whose surviving DocuSign envelope was sent well after the MCF
    'Sent for signature' status — the signature of a voided-and-resent envelope.

    resend_gap_days = docusign_sent_1 − s10 (calendar days). resend_suspected is
    True when both timestamps exist and the gap exceeds RESEND_GAP_DAYS. Quotes
    with no surviving DocuSign envelope cannot be flagged (the export omits voided
    envelopes), so this is a lower bound.
    """
    gap = (qt["docusign_sent_1"] - qt["s10"]).dt.total_seconds() / 86400
    qt["resend_gap_days"] = gap
    qt["resend_suspected"] = (
        qt["docusign_sent_1"].notna() & qt["s10"].notna() & (gap > RESEND_GAP_DAYS)
    )
    return qt


def _compute_phases(qt: pd.DataFrame) -> pd.DataFrame:
    """Add phase-span columns phase_cal_{k} (hours) / phase_biz_{k} (days), k=0..4.

    Computed from the phase endpoints in _PHASE_EP — used by both the HTML
    export and the Streamlit summary so the two stay in lock-step.
    """
    for k, (s_idx, e_idx) in enumerate(_PHASE_EP):
        start = qt[f"s{s_idx + 1:02d}"]
        end   = qt[f"s{e_idx + 1:02d}"]
        qt[f"phase_cal_{k}"] = (end - start).dt.total_seconds() / 3600
        qt[f"phase_biz_{k}"] = _biz_deltas(start, end)
    return qt


# ── Public entry point ────────────────────────────────────────────────────────

@_cache
def build_velocity_table(file_bytes: bytes) -> pd.DataFrame:
    """
    Accepts the raw bytes of O2C_v1.xlsx, returns a one-row-per-quote
    DataFrame with 16 step timestamps, 30 delta columns, and 7 flag columns.
    Cached by Streamlit on file_bytes hash.
    """
    xls = pd.ExcelFile(io.BytesIO(file_bytes))

    qh = _load_quote_history(xls)
    oh = _load_order_history(xls)
    ch = _load_contract_history(xls)
    ds = _load_docusign(xls)

    qt = _extract_quote_events(qh)
    qt = _join_order_history(qt, oh)
    qt = _join_contract_history(qt, ch)
    qt = _join_docusign(qt, ds)
    qt = _compute_flags(qt, qh)
    qt = _compute_deltas(qt)
    qt = _compute_phases(qt)
    qt = _compute_resend_flag(qt)

    return qt.reset_index(drop=True)


# ── HTML export ───────────────────────────────────────────────────────────────

def _serialize_quotes(df: pd.DataFrame) -> str:
    """Serialize the full quote table to a JSON array for the HTML dashboard."""
    records = []
    for _, row in df.iterrows():
        timestamps = []
        for i in range(1, 17):
            ts = row.get(f"s{i:02d}")
            timestamps.append(
                pd.Timestamp(ts).isoformat() if pd.notna(ts) else None
            )

        cal_h, biz_d = [], []
        for i in range(1, 16):
            c = row.get(f"delta_cal_h_{i}")
            b = row.get(f"delta_biz_d_{i}")
            cal_h.append(None if (c is None or (isinstance(c, float) and math.isnan(c))) else float(c))
            biz_d.append(None if (b is None or (isinstance(b, float) and math.isnan(b))) else float(b))

        qnum = row.get("quote_number")
        cn   = row.get("contract_number")
        opp  = row.get("opportunity_id")
        rs   = row.get("rework_stage")

        # ── Phase spans (precomputed in _compute_phases): Quote, DocuSign,
        #    Contract, Order, Full — read straight from columns ───────────────
        phase_cal_v: list = []
        phase_biz_v: list = []
        for k in range(len(_PHASE_EP)):
            c = row.get(f"phase_cal_{k}")
            b = row.get(f"phase_biz_{k}")
            phase_cal_v.append(None if (c is None or (isinstance(c, float) and math.isnan(c))) else float(c))
            phase_biz_v.append(None if (b is None or (isinstance(b, float) and math.isnan(b))) else float(b))

        records.append({
            "quote_id":         str(row.get("quote_id", "")),
            "quote_number":     f"Q{int(qnum)}" if pd.notna(qnum) else None,
            "opportunity":      str(opp) if pd.notna(opp) else None,
            "contract_number":  str(int(cn)) if pd.notna(cn) else None,
            "order_count":      int(row.get("order_count", 0)),
            "rework_flag":      bool(row.get("rework_flag", False)),
            "rework_stage":     str(rs) if pd.notna(rs) else None,
            "rework_count":     int(row.get("rework_count", 0)),
            "nsct_flag":        bool(row.get("nsct_flag", False)),
            "outcome":          str(row.get("outcome", "In Progress")),
            "multi_order_flag": bool(row.get("multi_order_flag", False)),
            "resend_suspected": bool(row.get("resend_suspected", False)),
            "resend_gap_days":  (None if pd.isna(row.get("resend_gap_days"))
                                 else float(row.get("resend_gap_days"))),
            "has_ds_sent":      bool(pd.notna(row.get("docusign_sent_1"))),
            "timestamps":       timestamps,
            # cal: calendar hours — JS divides by 24 for days, uses as-is for hours
            "cal":              cal_h,
            # biz: business days — JS multiplies by 8 for hours, uses as-is for days
            "biz":              biz_d,
            # phase spans: [Quote, DocuSign, Contract, Order, Full]
            "phase_cal":        phase_cal_v,
            "phase_biz":        phase_biz_v,
        })
    return json.dumps(records, separators=(",", ":"))


def build_standalone_html(df: pd.DataFrame) -> str:
    """
    Full interactive HTML dashboard in the same style as velocity_dashboard.html.
    Embeds all quote data as JSON. Client-side filters: Month (from s01),
    Rework, NSCT, Outcome. Toggles: Calendar/Business and Days/Hours.
    """
    quotes_json = _serialize_quotes(df)

    css = """
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,-apple-system,sans-serif;background:#f0f4f8;color:#1a202c;font-size:14px;line-height:1.5}
.wrap{max-width:1280px;margin:0 auto;padding:0 16px 60px}
header{background:#1e293b;color:#fff;padding:16px 24px;border-radius:0 0 10px 10px;margin-bottom:20px;box-shadow:0 2px 8px rgba(0,0,0,.25)}
header h1{font-size:18px;font-weight:700;letter-spacing:.01em}
.layout{display:flex;gap:20px;align-items:flex-start}
.sidebar{width:200px;flex-shrink:0;background:#fff;border-radius:10px;box-shadow:0 1px 3px rgba(0,0,0,.08);padding:18px 14px;position:sticky;top:16px}
.main{flex:1;min-width:0;overflow:hidden}
.sb-section{margin-bottom:16px}
.sb-section:last-child{margin-bottom:0}
.sb-lbl{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:#64748b;margin-bottom:6px;display:block}
.sb-toggle{display:flex;border:1px solid #e2e8f0;border-radius:6px;overflow:hidden}
.sb-btn{flex:1;background:none;border:none;padding:7px 4px;font-size:11px;font-weight:500;color:#64748b;cursor:pointer;transition:.15s;text-align:center}
.sb-btn.active{background:#1e293b;color:#fff}
.sb-chips{display:flex;flex-wrap:wrap;gap:4px;margin-top:4px}
.sb-divider{border:none;border-top:1px solid #f1f5f9;margin:12px 0}
.tabs{display:flex;border-bottom:2px solid #e2e8f0;margin-bottom:20px;gap:4px}
.tab{background:none;border:none;border-bottom:2px solid transparent;margin-bottom:-2px;padding:10px 22px;cursor:pointer;font-size:14px;font-weight:500;color:#64748b;transition:.15s}
.tab.active{color:#2563eb;border-bottom-color:#2563eb}
.dv-toggle{display:inline-flex;border:1px solid #e2e8f0;border-radius:7px;overflow:hidden;margin-bottom:14px}
.dv-btn{background:none;border:none;padding:8px 18px;font-size:12px;font-weight:600;color:#64748b;cursor:pointer;transition:.15s}
.dv-btn.active{background:#2563eb;color:#fff}
.filters{background:#fff;padding:16px 20px;border-radius:10px;margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,.08);display:flex;flex-wrap:wrap;gap:20px;align-items:center}
.fg{display:flex;align-items:center;gap:8px}
.fg label{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:#64748b;white-space:nowrap}
.chips{display:flex;gap:4px;flex-wrap:wrap}
.chip{background:#f1f5f9;border:1px solid #e2e8f0;border-radius:20px;padding:4px 13px;font-size:12px;font-weight:500;color:#475569;cursor:pointer;transition:.15s;white-space:nowrap}
.chip:hover{background:#dde4ed}
.chip.on{background:#2563eb;border-color:#2563eb;color:#fff}
.callout{background:#fffbeb;border:1px solid #fbbf24;border-left:4px solid #f59e0b;border-radius:6px;padding:14px 18px;margin-bottom:16px}
.callout h4{font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:#92400e;margin-bottom:10px}
.callout p{font-size:13px;color:#78350f;margin-bottom:8px}
.co-tbl{width:100%;border-collapse:collapse;font-size:13px}
.co-tbl th{text-align:left;padding:3px 12px;color:#92400e;font-weight:600;font-size:11px;text-transform:uppercase}
.co-tbl td{padding:4px 12px;border-top:1px solid #fde68a}
.co-tbl .pos{color:#dc2626;font-weight:700}
.co-tbl .neg{color:#16a34a;font-weight:700}
.tbl-wrap{background:#fff;border-radius:10px;box-shadow:0 1px 3px rgba(0,0,0,.08);overflow-x:auto}
.vtbl{width:100%;border-collapse:collapse}
.vtbl thead tr{background:#1e293b}
.vtbl thead th{padding:11px 16px;text-align:left;font-size:11px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:#94a3b8;white-space:nowrap}
.vtbl thead th.r{text-align:right}
.vtbl tbody tr{border-bottom:1px solid #f1f5f9}
.vtbl tbody tr:last-child{border-bottom:none}
.vtbl tbody tr:hover{background:#f8fafc}
.vtbl td{padding:10px 16px;font-size:13px;color:#374151}
.vtbl td.pair{font-weight:600;color:#1e293b}
.vtbl td.num{text-align:right;font-variant-numeric:tabular-nums;color:#1e293b}
.vtbl td.muted{text-align:right;color:#94a3b8}
.vtbl td.rwd{text-align:right;font-weight:700;color:#dc2626}
.vtbl td.rwd.better{color:#16a34a}
.search-box{background:#fff;padding:18px 20px;border-radius:10px;box-shadow:0 1px 3px rgba(0,0,0,.08);margin-bottom:20px}
.search-row{display:flex;gap:8px}
.search-row input{flex:1;padding:10px 14px;border:1px solid #d1d5db;border-radius:7px;font-size:14px;outline:none;transition:.15s}
.search-row input:focus{border-color:#2563eb;box-shadow:0 0 0 3px rgba(37,99,235,.12)}
.search-row button{padding:10px 22px;background:#2563eb;color:#fff;border:none;border-radius:7px;cursor:pointer;font-size:14px;font-weight:600}
.search-row button:hover{background:#1d4ed8}
.qcard{background:#fff;border-radius:10px;box-shadow:0 1px 3px rgba(0,0,0,.08);margin-bottom:20px;overflow:hidden}
.qcard-hd{background:#f8fafc;border-bottom:1px solid #e2e8f0;padding:12px 18px;display:flex;flex-wrap:wrap;gap:10px;align-items:center}
.qcard-title{font-size:15px;font-weight:700;color:#1e293b}
.badge{display:inline-flex;align-items:center;padding:3px 11px;border-radius:20px;font-size:11px;font-weight:700;letter-spacing:.04em;text-transform:uppercase}
.b-acc{background:#dcfce7;color:#166534}
.b-rej{background:#fee2e2;color:#991b1b}
.b-den{background:#fce7f3;color:#9d174d}
.b-inp{background:#dbeafe;color:#1e40af}
.b-rw{background:#fee2e2;color:#991b1b}
.b-nsct{background:#ede9fe;color:#5b21b6}
.b-mo{background:#fef3c7;color:#92400e}
.b-opp{background:#f1f5f9;color:#475569;text-transform:none;font-weight:500;font-size:11px;letter-spacing:0}
.qcard-body{overflow-x:auto}
.ttbl{width:100%;border-collapse:collapse}
.ttbl th{background:#f8fafc;padding:7px 16px;text-align:left;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.06em;color:#64748b;border-bottom:1px solid #e2e8f0}
.ttbl th.r{text-align:right}
.ttbl td{padding:8px 16px;font-size:13px;border-bottom:1px solid #f8fafc}
.ttbl tr:last-child td{border-bottom:none}
.ttbl .sn{color:#94a3b8;font-size:11px;width:28px}
.ttbl .sl{font-weight:500}
.ttbl .ts{font-family:ui-monospace,monospace;font-size:12px;color:#374151}
.ttbl .el{text-align:right;font-variant-numeric:tabular-nums}
.ttbl .null{color:#94a3b8}
.ttbl .pend{color:#d97706;font-style:italic}
.no-res{text-align:center;color:#94a3b8;padding:48px;font-size:15px}
.hidden{display:none!important}
.summary{background:#fff;border-radius:10px;box-shadow:0 1px 3px rgba(0,0,0,.08);margin-top:20px;overflow:hidden}
.summary-hdr{background:#1e293b;color:#94a3b8;padding:10px 16px;font-size:11px;font-weight:600;letter-spacing:.06em;text-transform:uppercase}
.chart-card{background:#fff;border-radius:10px;box-shadow:0 1px 3px rgba(0,0,0,.08);margin-top:20px;overflow:hidden}
.chart-hdr{background:#1e293b;color:#94a3b8;padding:10px 16px;font-size:11px;font-weight:600;letter-spacing:.06em;text-transform:uppercase}
.chart-body{padding:16px 12px 12px}
.stat-select{padding:8px 10px;border:1px solid #d1d5db;border-radius:7px;font-size:13px;margin-bottom:12px;min-width:280px;background:#fff;color:#1a202c}
.vtbl th[title]{cursor:help;text-decoration:underline dotted #64748b}
"""

    # Labels injected from the Python constants above so the HTML export and the
    # Streamlit app stay in lock-step (single source of truth).
    label_js = (
        f"const STEP_LABELS={json.dumps(STEP_LABELS)};\n"
        f"const PAIR_LABELS={json.dumps(PAIR_LABELS)};\n"
        f"const PHASE_LABELS={json.dumps(PHASE_LABELS)};\n"
        f"const DEST_LABELS={json.dumps([STEP_LABELS[i + 1] for i in range(15)])};\n"
    )
    js = label_js + r"""
const OBJECT_PAIRS={all:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],quote:[0,1,2,3,4,5,6,7],docusign:[8,9,10],contract:[11],order:[12,13,14]};
/* object color palettes — indexed by pair (0-14) and step (0-15) */
const OBJ_COLORS=['#3b82f6','#3b82f6','#3b82f6','#3b82f6','#3b82f6','#3b82f6','#3b82f6','#3b82f6','#7c3aed','#7c3aed','#7c3aed','#0891b2','#059669','#059669','#059669'];
const STEP_COLORS=['#3b82f6','#3b82f6','#3b82f6','#3b82f6','#3b82f6','#3b82f6','#3b82f6','#3b82f6','#3b82f6','#7c3aed','#7c3aed','#7c3aed','#0891b2','#059669','#059669','#059669'];

const state={mode:'cal',unit:'days',rework:'all',nsct:'all',outcome:'all',month:'all',object:'all',dataView:'table',resend:'all',statPair:8};

function avg(arr){const v=arr.filter(x=>x!==null&&!isNaN(x));return v.length?v.reduce((a,b)=>a+b,0)/v.length:null;}
function med(arr){const v=[...arr.filter(x=>x!==null&&!isNaN(x))].sort((a,b)=>a-b);if(!v.length)return null;const m=Math.floor(v.length/2);return v.length%2?v[m]:(v[m-1]+v[m])/2;}
/* linear-interpolation percentile (matches pandas/numpy default) */
function percentile(sorted,p){if(!sorted.length)return null;const idx=(sorted.length-1)*p,lo=Math.floor(idx),hi=Math.ceil(idx);return lo===hi?sorted[lo]:sorted[lo]+(sorted[hi]-sorted[lo])*(idx-lo);}
function stdev(arr){if(!arr.length)return null;const m=arr.reduce((a,b)=>a+b,0)/arr.length;return Math.sqrt(arr.reduce((a,b)=>a+(b-m)*(b-m),0)/arr.length);}
/* per-step-pair stats over a filtered set (excludes negative deltas) */
function pairStats(filtered,i){
  const vals=filtered.map(q=>getVal(q,i)).filter(x=>x!==null&&x>=0).sort((a,b)=>a-b);
  if(!vals.length)return null;
  return{n:vals.length,min:vals[0],max:vals[vals.length-1],mean:vals.reduce((a,b)=>a+b,0)/vals.length,median:percentile(vals,.5),std:stdev(vals),p90:percentile(vals,.9),p95:percentile(vals,.95),vals:vals};
}
function fmt(v,d=1){return v===null||v===undefined?'—':v.toFixed(d);}
function fmtTs(iso){if(!iso)return null;const d=new Date(iso);return d.toLocaleDateString('en-US',{month:'short',day:'numeric',year:'numeric'})+' '+d.toLocaleTimeString('en-US',{hour:'2-digit',minute:'2-digit'});}
function escHtml(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function tsMonth(iso){return iso?iso.substring(0,7):null;}
function monthLabel(ym){if(!ym)return'?';const[y,m]=ym.split('-');return['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][parseInt(m)-1]+' '+y;}

/* unit conversion: cal stored as hours, biz stored as days */
function getVal(q,i){
  const raw=state.mode==='cal'?q.cal[i]:q.biz[i];
  if(raw===null||raw===undefined||isNaN(raw))return null;
  if(state.mode==='cal')return state.unit==='hours'?raw:raw/24;
  return state.unit==='hours'?raw*8:raw;
}
function unitSuffix(){return state.unit==='hours'?'h':'d';}
function unitName(){return(state.mode==='cal'?'Calendar ':'Business ')+(state.unit==='days'?'Days':'Hours');}

/* phase-span unit conversion (same logic as getVal but reads phase_cal/phase_biz) */
function getPhaseVal(q,i){
  const raw=state.mode==='cal'?q.phase_cal[i]:q.phase_biz[i];
  if(raw===null||raw===undefined||isNaN(raw))return null;
  if(state.mode==='cal')return state.unit==='hours'?raw:raw/24;
  return state.unit==='hours'?raw*8:raw;
}

function setObjectFilter(btn){
  const grp=btn.closest('.sb-chips,.chips');
  grp.querySelectorAll('.chip').forEach(c=>c.classList.remove('on'));
  btn.classList.add('on');
  state.object=btn.dataset.v;
  renderViewA();
}

function renderSummary(filtered){
  const u=unitSuffix(),un=unitName();
  /* compute stats for each of the 4 base phases (indices 0–3) */
  const pd=[0,1,2,3].map(function(i){
    const vals=filtered.map(function(q){return getPhaseVal(q,i);}).filter(function(x){return x!==null;});
    return{lbl:PHASE_LABELS[i],a:avg(vals),md:med(vals),n:vals.length};
  });

  function phaseRow(d,style){
    return'<tr'+(style?' style="'+style+'"':'')+'>'
      +'<td class="pair">'+escHtml(d.lbl)+'</td>'
      +'<td class="'+(d.a===null?'muted':'num')+'">'+fmt(d.a)+'</td>'
      +'<td class="'+(d.md===null?'muted':'num')+'">'+fmt(d.md)+'</td>'
      +'<td class="num">'+d.n+'</td></tr>';
  }
  function sumRow(lbl,phaseIdxs,style){
    /* Avg = sum of individual phase avgs; Median / n not meaningful for a sum */
    const parts=phaseIdxs.map(function(i){return pd[i].a;});
    const a=parts.every(function(v){return v!==null;})?parts.reduce(function(s,v){return s+v;},0):null;
    return'<tr style="'+(style||'')+'border-top:2px solid #e2e8f0;font-weight:600">'
      +'<td class="pair">'+escHtml(lbl)+'</td>'
      +'<td class="'+(a===null?'muted':'num')+'">'+fmt(a)+'</td>'
      +'<td class="muted">—</td>'
      +'<td class="muted">—</td></tr>';
  }

  let html='<div class="summary"><div class="summary-hdr">End-to-End Summary — '+un+'</div>'
    +'<table class="vtbl"><thead><tr><th>Phase</th><th class="r">Avg ('+un+')</th>'
    +'<th class="r">Median</th><th class="r">n</th></tr></thead><tbody>';
  /* individual phases */
  html+=phaseRow(pd[0]);
  html+=phaseRow(pd[1]);
  html+=phaseRow(pd[2]);
  /* subtotal: quote creation → contract activated (sum of phases 0+1+2) */
  html+=sumRow('Created → Contract Activated (subtotal)',[0,1,2],'background:#f8fafc;');
  /* order phase */
  html+=phaseRow(pd[3]);
  /* full cycle: sum of all 4 phase avgs */
  html+=sumRow('Full Cycle — Created → Deployed',[0,1,2,3],'');
  html+='</tbody></table></div>';
  document.getElementById('summary-a').innerHTML=html;
}

/* ── Chart rendering ─────────────────────────────────────────────────────── */
function renderBarChart(allRows,u){
  /* wider label area so full pair names fit without truncation */
  const LW=300,BW=500,VW=80,TW=LW+BW+VW;
  const rowH=28,gap=3,padT=14,padB=10;
  const H=padT+15*(rowH+gap)+padB;
  const avgs=allRows.map(r=>r.a);
  const maxV=Math.max(...avgs.filter(x=>x!==null),0.001);
  let s=`<svg viewBox="0 0 ${TW} ${H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;display:block" font-family="system-ui,sans-serif">`;
  allRows.forEach((r,i)=>{
    const y=padT+i*(rowH+gap);
    const bw=r.a!==null?(r.a/maxV)*BW:0;
    const col=OBJ_COLORS[r.pairIdx];
    const warn=r.pairIdx===14?' ⚠':'';
    /* row background */
    s+=`<rect x="0" y="${y}" width="${TW}" height="${rowH}" fill="${i%2===0?'#f8fafc':'#fff'}"/>`;
    /* full label — no truncation; SVG clips via viewBox if somehow too long */
    s+=`<text x="${LW-10}" y="${y+rowH/2+4}" text-anchor="end" font-size="10" fill="#374151">${escHtml(r.lbl)}${warn}</text>`;
    /* bar */
    if(bw>0)s+=`<rect x="${LW}" y="${y+6}" width="${bw}" height="${rowH-12}" fill="${col}" rx="3" opacity="0.85"/>`;
    /* value */
    s+=`<text x="${LW+bw+6}" y="${y+rowH/2+4}" font-size="11" fill="${r.a!==null?'#1e293b':'#94a3b8'}" font-weight="600">${r.a!==null?fmt(r.a)+u:'—'}</text>`;
  });
  s+='</svg>';
  return s;
}

function lerp(a,b,t){return Math.round(a+(b-a)*t);}
function heatColor(t){
  /* cool blue → yellow → hot red  (t: 0..1) */
  if(t<0.5){const s=t*2;return`rgb(${lerp(191,253,s)},${lerp(219,224,s)},${lerp(254,71,s)})`;}
  const s=(t-0.5)*2;return`rgb(${lerp(253,220,s)},${lerp(224,38,s)},${lerp(71,38,s)})`;
}
function heatTextColor(t){return t>0.55?'#fff':'#1e293b';}

function renderHeatmap(allRows,u){
  /* Compact proportional bar — no external labels.
     Each segment shows % of total; hover reveals step name + value via SVG <title>. */
  const W=860,padL=20,padR=20,barW=W-padL-padR;
  const barY=14,barH=68,H=112;
  const avgs=allRows.map(r=>r.a!==null?r.a:0);
  const total=avgs.reduce((a,b)=>a+b,0)||1;
  const nonNull=allRows.map(r=>r.a).filter(x=>x!==null);
  const minV=nonNull.length?Math.min(...nonNull):0;
  const maxV=nonNull.length?Math.max(...nonNull):1;
  const range=maxV-minV||1;

  const xs=[padL];
  avgs.forEach(v=>xs.push(xs[xs.length-1]+(v/total)*barW));

  const phases=[
    {s:0,e:8,col:'#3b82f6',lbl:'Quote (1–8)'},
    {s:8,e:11,col:'#7c3aed',lbl:'DocuSign (9–11)'},
    {s:11,e:12,col:'#0891b2',lbl:'Contract (12)'},
    {s:12,e:15,col:'#059669',lbl:'Order (13–15)'}
  ];

  let s=`<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;display:block;cursor:default" font-family="system-ui,sans-serif">`;

  /* ── segments — each wrapped in <g> with <title> for hover tooltip ── */
  allRows.forEach((r,i)=>{
    const x1=xs[i],x2=xs[i+1],sw=x2-x1;
    const t=r.a!==null?(r.a-minV)/range:0;
    const fill=r.a!==null?heatColor(t):'#f1f5f9';
    const tc=r.a!==null?heatTextColor(t):'#94a3b8';
    const pct=r.a!==null?((r.a/total)*100).toFixed(1):'0.0';
    const warn=i===14?' ⚠':'';
    const tipLbl=DEST_LABELS[i]+warn;
    const tipVal=r.a!==null?fmt(r.a)+u+' ('+pct+'%)':'no data';
    /* tooltip: browser shows this on hover */
    s+=`<g><title>Pair ${i+1}: ${tipLbl} — ${tipVal}</title>`;
    s+=`<rect x="${x1}" y="${barY}" width="${sw}" height="${barH}" fill="${fill}"/>`;
    /* phase-colour top strip */
    s+=`<rect x="${x1}" y="${barY}" width="${sw}" height="4" fill="${OBJ_COLORS[i]}" opacity="0.65"/>`;
    /* white divider */
    if(i>0)s+=`<line x1="${x1}" y1="${barY}" x2="${x1}" y2="${barY+barH}" stroke="#fff" stroke-width="1.5"/>`;
    /* step number (always shown) */
    const numFs=sw>26?13:sw>14?10:8;
    s+=`<text x="${(x1+x2)/2}" y="${barY+barH/2+1}" text-anchor="middle" font-size="${numFs}" font-weight="700" fill="${tc}">${i+1}</text>`;
    /* percentage inside segment — only if wide enough to read */
    if(sw>46)s+=`<text x="${(x1+x2)/2}" y="${barY+barH/2+15}" text-anchor="middle" font-size="9" fill="${tc}" opacity="0.9">${pct}%</text>`;
    s+=`</g>`;
  });

  /* bar outline */
  s+=`<rect x="${padL}" y="${barY}" width="${barW}" height="${barH}" fill="none" stroke="#cbd5e1" stroke-width="1"/>`;

  /* ── legend row ── */
  const lgY=H-12;
  const lgW=80,lgH=6,swatchSteps=16;
  for(let i=0;i<swatchSteps;i++){
    s+=`<rect x="${padL+i*(lgW/swatchSteps)}" y="${lgY}" width="${lgW/swatchSteps+0.5}" height="${lgH}" fill="${heatColor(i/(swatchSteps-1))}"/>`;
  }
  s+=`<rect x="${padL}" y="${lgY}" width="${lgW}" height="${lgH}" fill="none" stroke="#e2e8f0" stroke-width="0.5"/>`;
  s+=`<text x="${padL}" y="${lgY-3}" font-size="8" fill="#94a3b8">Less</text>`;
  s+=`<text x="${padL+lgW}" y="${lgY-3}" text-anchor="end" font-size="8" fill="#94a3b8">More</text>`;
  let plx=padL+lgW+20;
  phases.forEach(ph=>{
    s+=`<rect x="${plx}" y="${lgY}" width="${lgH+1}" height="${lgH}" fill="${ph.col}" rx="1"/>`;
    s+=`<text x="${plx+lgH+4}" y="${lgY+lgH}" font-size="9" fill="#374151">${ph.lbl}</text>`;
    plx+=130;
  });
  s+=`<text x="${padL+barW}" y="${lgY+lgH}" text-anchor="end" font-size="9" fill="#94a3b8">Total: ${total>0?fmt(total)+u:'—'} avg</text>`;

  s+='</svg>';
  return s;
}

function renderCharts(allRows,u){
  document.getElementById('charts-a').innerHTML=
    '<div class="chart-card"><div class="chart-hdr">Avg Time per Step Pair — '+unitName()+'</div>'
    +'<div class="chart-body">'+renderBarChart(allRows,u)+'</div></div>'
    +'<div class="chart-card"><div class="chart-hdr">Time Distribution — % of Avg Cycle (hover each segment for step name &amp; value)</div>'
    +'<div class="chart-body">'+renderHeatmap(allRows,u)+'</div></div>';
}

function applyFilters(qs,s){
  s=s||state;
  return qs.filter(q=>{
    if(s.rework==='rework'&&!q.rework_flag)return false;
    if(s.rework==='clean'&&q.rework_flag)return false;
    if(s.nsct==='nsct'&&!q.nsct_flag)return false;
    if(s.nsct==='non_nsct'&&q.nsct_flag)return false;
    if(s.outcome!=='all'&&q.outcome!==s.outcome)return false;
    if(s.month&&s.month!=='all'&&tsMonth(q.timestamps[0])!==s.month)return false;
    if(s.resend==='exclude'&&q.resend_suspected)return false;
    return true;
  });
}

function setMode(m){
  state.mode=m;
  document.getElementById('btn-cal').classList.toggle('active',m==='cal');
  document.getElementById('btn-biz').classList.toggle('active',m==='biz');
  renderViewA();renderViewC();const sv=document.getElementById('si').value.trim();if(sv)renderViewB(sv);
}
function setUnit(u){
  state.unit=u;
  document.getElementById('btn-days').classList.toggle('active',u==='days');
  document.getElementById('btn-hours').classList.toggle('active',u==='hours');
  renderViewA();renderViewC();const sv=document.getElementById('si').value.trim();if(sv)renderViewB(sv);
}
function setView(v){
  ['A','B','C'].forEach(function(x){
    document.getElementById('view-'+x.toLowerCase()).classList.toggle('hidden',x!==v);
    document.getElementById('tab-'+x.toLowerCase()).classList.toggle('active',x===v);
  });
}
function setFilter(key,btn){
  const grp=btn.closest('.sb-chips,.chips');
  grp.querySelectorAll('.chip').forEach(c=>c.classList.remove('on'));
  btn.classList.add('on');state[key]=btn.dataset.v;renderViewA();renderViewC();
}
function setStatPair(v){state.statPair=+v;renderStatHist(applyFilters(QUOTES),state.statPair);}
function setDataView(v){
  state.dataView=v;
  document.getElementById('tbl-a').classList.toggle('hidden',v!=='table');
  document.getElementById('charts-a').classList.toggle('hidden',v!=='chart');
  document.getElementById('dv-table').classList.toggle('active',v==='table');
  document.getElementById('dv-chart').classList.toggle('active',v==='chart');
}

function renderViewA(){
  const filtered=applyFilters(QUOTES);
  const isRework=state.rework==='rework';
  const cleanQ=isRework?applyFilters(QUOTES,{...state,rework:'clean'}):null;
  const u=unitSuffix(),un=unitName();
  /* all 15 rows computed; then scoped to the active object filter */
  const allRows=PAIR_LABELS.map((lbl,i)=>{
    const vals=filtered.map(q=>getVal(q,i)).filter(x=>x!==null);
    const a=avg(vals),md=med(vals),n=vals.length;
    let delta=null;
    if(isRework&&cleanQ){const cv=cleanQ.map(q=>getVal(q,i)).filter(x=>x!==null);const ca=avg(cv);if(a!==null&&ca!==null)delta=a-ca;}
    return{lbl,a,md,n,delta,pairIdx:i};
  });
  const vis=OBJECT_PAIRS[state.object]||OBJECT_PAIRS.all;
  const rows=allRows.filter(r=>vis.includes(r.pairIdx));
  const callout=document.getElementById('callout-rework');
  if(isRework){
    const slower=rows.filter(r=>r.delta!==null&&r.delta>0);
    if(slower.length){
      callout.innerHTML='<h4>Rework Cost — Steps Where Rework Quotes Are Slower</h4><p>Compared to clean-path quotes under the same filters.</p>'
        +'<table class="co-tbl"><tr><th>Step Pair</th><th>Rework Avg</th><th>Clean Avg</th><th>Delta</th></tr>'
        +slower.map(r=>`<tr><td>${escHtml(r.lbl)}</td><td>${fmt(r.a)}${u}</td><td>${fmt(r.a-r.delta)}${u}</td><td class="pos">+${fmt(r.delta)}${u}</td></tr>`).join('')+'</table>';
      callout.classList.remove('hidden');
    }else{
      callout.innerHTML='<h4>Rework Cost</h4><p>No step pairs where rework quotes average more time than clean-path quotes.</p>';
      callout.classList.remove('hidden');
    }
  }else{callout.classList.add('hidden');}
  const xhdr=isRework?'<th class="r">vs. Clean Path</th>':'';
  let html=`<table class="vtbl"><thead><tr><th>Step Pair</th><th class="r">Avg (${un})</th><th class="r">Median</th><th class="r">n</th>${xhdr}</tr></thead><tbody>`;
  rows.forEach(r=>{
    const warn=r.pairIdx===14?' <span title="Manually set; may lag actual install date">⚠️</span>':'';
    const xtd=isRework?(r.delta===null?'<td class="muted">—</td>':r.delta>0?`<td class="rwd">+${fmt(r.delta)}${u}</td>`:`<td class="rwd better">${fmt(r.delta)}${u}</td>`):'';
    html+=`<tr><td class="pair">${escHtml(r.lbl)}${warn}</td><td class="${r.a===null?'muted':'num'}">${fmt(r.a)}</td><td class="${r.md===null?'muted':'num'}">${fmt(r.md)}</td><td class="num">${r.n}</td>${xtd}</tr>`;
  });
  html+='</tbody></table>';
  document.getElementById('tbl-a').innerHTML=html;
  renderCharts(allRows,unitSuffix());   // always full 15 pairs, regardless of object filter
  renderSummary(filtered);
}

/* ── View C: Statistical Analysis (resend callout + stats table + histogram) ── */
const _STAT_HELP={n:'Quotes with a valid, non-negative duration (after filters).',Min:'Shortest observed duration.',Max:'Longest observed duration — the outlier ceiling.',Mean:'Average; pulled upward by long tails.',Median:'Middle value; half faster, half slower — robust to outliers.',Std:'Standard deviation — spread around the mean.',p90:'90% of quotes finish this step within this time.',p95:'95% finish within this time — highlights the slow tail.'};

function renderResendCallout(){
  const co=document.getElementById('callout-resend');
  const flagged=QUOTES.filter(q=>q.resend_suspected);
  const nSent=QUOTES.filter(q=>q.has_ds_sent).length;
  if(!flagged.length){co.classList.add('hidden');return;}
  const gaps=flagged.map(q=>q.resend_gap_days).filter(x=>x!==null&&!isNaN(x)).sort((a,b)=>a-b);
  const gMed=percentile(gaps,.5),gMax=gaps[gaps.length-1],gSum=gaps.reduce((a,b)=>a+b,0);
  co.innerHTML='<h4>Resend Opportunity</h4><p><strong>'+flagged.length+' quotes ('+Math.round(flagged.length/Math.max(nSent,1)*100)+'% of DocuSign-sent)</strong> show a voided/resent envelope. The envelope was actually sent a median of <strong>'+gMed.toFixed(1)+' days</strong> (up to '+Math.round(gMax)+') after the quote was marked ‘Sent for signature’ — a total of <strong>'+Math.round(gSum)+' days</strong> of avoidable delay. Use the sidebar <strong>Resend-suspected → Exclude</strong> to see clean-process metrics.</p>';
  co.classList.remove('hidden');
}

function renderStatHist(filtered,i){
  const el=document.getElementById('stat-hist');
  const s=pairStats(filtered,i);
  if(!s){el.innerHTML='<div class="no-res">No data for this step pair under the current filters.</div>';return;}
  const vals=s.vals,u=unitSuffix();
  const W=860,padL=48,padR=20,padT=16,padB=42,H=320,bins=30;
  const min=0,max=s.max||1,bw=(max-min)/bins||1;
  const counts=new Array(bins).fill(0);
  vals.forEach(v=>{let b=Math.floor((v-min)/bw);if(b<0)b=0;if(b>=bins)b=bins-1;counts[b]++;});
  const maxC=Math.max.apply(null,counts.concat([1]));
  const plotW=W-padL-padR,plotH=H-padT-padB;
  let svg=`<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;display:block" font-family="system-ui,sans-serif">`;
  for(let g=0;g<=4;g++){const yv=Math.round(maxC*g/4);const y=padT+plotH-(plotH*g/4);svg+=`<line x1="${padL}" y1="${y}" x2="${W-padR}" y2="${y}" stroke="#f1f5f9"/><text x="${padL-6}" y="${y+3}" text-anchor="end" font-size="9" fill="#94a3b8">${yv}</text>`;}
  counts.forEach((c,b)=>{if(!c)return;const x=padL+(plotW*b/bins);const w=Math.max(plotW/bins-1,1);const bh=plotH*c/maxC;const y=padT+plotH-bh;svg+=`<rect x="${x}" y="${y}" width="${w}" height="${bh}" fill="#3b82f6" opacity="0.8" rx="1"><title>${c} quotes</title></rect>`;});
  for(let t=0;t<=6;t++){const xv=min+(max-min)*t/6;const x=padL+plotW*t/6;svg+=`<text x="${x}" y="${H-padB+16}" text-anchor="middle" font-size="9" fill="#94a3b8">${xv.toFixed(1)}</text>`;}
  svg+=`<text x="${padL+plotW/2}" y="${H-6}" text-anchor="middle" font-size="10" fill="#64748b">Duration (${unitName()})</text>`;
  [['median',s.median],['p90',s.p90],['p95',s.p95]].forEach(pr=>{const lb=pr[0],val=pr[1];if(val==null)return;const x=padL+plotW*(val-min)/((max-min)||1);svg+=`<line x1="${x}" y1="${padT}" x2="${x}" y2="${padT+plotH}" stroke="#dc2626" stroke-width="1" stroke-dasharray="4 3"><title>${lb}: ${val.toFixed(1)}${u}</title></line><text x="${x}" y="${padT-3}" text-anchor="middle" font-size="8" fill="#dc2626">${lb}</text>`;});
  svg+='</svg>';
  el.innerHTML=svg;
}

function renderViewC(){
  const filtered=applyFilters(QUOTES),u=unitSuffix();
  renderResendCallout();
  let h='<table class="vtbl"><thead><tr><th>Step Pair</th>'
    +'<th class="r" title="'+_STAT_HELP.n+'">n</th>'
    +'<th class="r" title="'+_STAT_HELP.Min+'">Min ('+u+')</th>'
    +'<th class="r" title="'+_STAT_HELP.Max+'">Max ('+u+')</th>'
    +'<th class="r" title="'+_STAT_HELP.Mean+'">Mean ('+u+')</th>'
    +'<th class="r" title="'+_STAT_HELP.Median+'">Median ('+u+')</th>'
    +'<th class="r" title="'+_STAT_HELP.Std+'">Std ('+u+')</th>'
    +'<th class="r" title="'+_STAT_HELP.p90+'">p90 ('+u+')</th>'
    +'<th class="r" title="'+_STAT_HELP.p95+'">p95 ('+u+')</th></tr></thead><tbody>';
  PAIR_LABELS.forEach((lbl,i)=>{
    const s=pairStats(filtered,i);
    if(s){h+='<tr><td class="pair">'+escHtml(lbl)+'</td><td class="num">'+s.n+'</td><td class="num">'+fmt(s.min)+'</td><td class="num">'+fmt(s.max)+'</td><td class="num">'+fmt(s.mean)+'</td><td class="num">'+fmt(s.median)+'</td><td class="num">'+fmt(s.std)+'</td><td class="num">'+fmt(s.p90)+'</td><td class="num">'+fmt(s.p95)+'</td></tr>';}
    else{h+='<tr><td class="pair">'+escHtml(lbl)+'</td><td class="num">0</td><td class="muted">—</td><td class="muted">—</td><td class="muted">—</td><td class="muted">—</td><td class="muted">—</td><td class="muted">—</td><td class="muted">—</td></tr>';}
  });
  h+='</tbody></table>';
  document.getElementById('stats-c').innerHTML=h;
  renderStatHist(filtered,state.statPair);
}

function doSearch(){const v=document.getElementById('si').value.trim();renderViewB(v);}
function renderViewB(search){
  const el=document.getElementById('sr');
  if(!search){el.innerHTML='';return;}
  const s=search.toLowerCase();
  const matches=QUOTES.filter(q=>(q.quote_number&&q.quote_number.toLowerCase()===s)||(q.opportunity&&q.opportunity.toLowerCase()===s));
  if(!matches.length){el.innerHTML=`<div class="no-res">No quotes found for <strong>${escHtml(search)}</strong></div>`;return;}
  matches.sort((a,b)=>(a.quote_number||'').localeCompare(b.quote_number||''));
  el.innerHTML=matches.map(q=>buildCard(q)).join('');
}
function outcomeBadge(o){const map={Accepted:'b-acc',Rejected:'b-rej',Denied:'b-den','In Progress':'b-inp'};return`<span class="badge ${map[o]||'b-inp'}">${escHtml(o)}</span>`;}
function buildCard(q){
  let badges=outcomeBadge(q.outcome);
  if(q.rework_flag)badges+=` <span class="badge b-rw">Rework${q.rework_stage?' · '+q.rework_stage:''}</span>`;
  if(q.nsct_flag)badges+=` <span class="badge b-nsct">NSCT</span>`;
  if(q.multi_order_flag)badges+=` <span class="badge b-mo">Multi-Order (${q.order_count})</span>`;
  if(q.opportunity)badges+=` <span class="badge b-opp">${escHtml(q.opportunity)}</span>`;
  let trows='';
  for(let i=0;i<16;i++){
    const ts=q.timestamps[i],isDS=i>=9&&i<=11,isClose=i===15,hasC=!!q.contract_number;
    const tsHtml=ts?`<span class="ts">${fmtTs(ts)}</span>`:(isDS&&hasC?'<span class="pend">Pending</span>':'<span class="null">—</span>');
    let elHtml='<span class="null">—</span>';
    if(i>0){const v=getVal(q,i-1),u=unitSuffix();if(v!==null)elHtml=`${v.toFixed(1)} ${u}`;else if(isDS&&hasC&&!q.timestamps[i-1])elHtml='<span class="pend">Pending</span>';}
    const lbl=isClose?`${escHtml(STEP_LABELS[i])} <span title="Manually set; may lag actual install date">⚠️</span>`:escHtml(STEP_LABELS[i]);
    const mo=(q.multi_order_flag&&i>=13)?` <span title="Excluded from averages for multi-order quotes" style="color:#d97706">⚠️</span>`:'';
    trows+=`<tr><td class="sn">${i+1}</td><td class="sl">${lbl}${mo}</td><td>${tsHtml}</td><td class="el">${elHtml}</td></tr>`;
  }
  return`<div class="qcard"><div class="qcard-hd"><span class="qcard-title">Quote ${escHtml(q.quote_number||q.quote_id)}</span>${badges}</div><div class="qcard-body"><table class="ttbl"><thead><tr><th>#</th><th>Event</th><th>Timestamp</th><th class="r">Elapsed (${unitName()})</th></tr></thead><tbody>${trows}</tbody></table></div></div>`;
}

(function(){
  const months=[...new Set(QUOTES.map(q=>tsMonth(q.timestamps[0])).filter(Boolean))].sort();
  const cont=document.getElementById('chips-month');
  months.forEach(ym=>{
    const btn=document.createElement('button');
    btn.className='chip';btn.dataset.v=ym;btn.textContent=monthLabel(ym);
    btn.onclick=function(){setFilter('month',this);};
    cont.appendChild(btn);
  });
  const sel=document.getElementById('stat-sel');
  PAIR_LABELS.forEach((lbl,i)=>{const o=document.createElement('option');o.value=i;o.textContent=lbl;if(i===state.statPair)o.selected=true;sel.appendChild(o);});
})();
renderViewA();
renderViewC();
"""

    body = (
        '<div class="wrap">\n'
        '  <header><h1>Order-to-Cash Velocity Dashboard</h1></header>\n'
        '  <div class="layout">\n'
        # ── Sidebar ──────────────────────────────────────────────────────────
        '    <div class="sidebar">\n'
        '      <div class="sb-section">\n'
        '        <span class="sb-lbl">Time Basis</span>\n'
        '        <div class="sb-toggle">\n'
        "          <button id=\"btn-cal\"  class=\"sb-btn active\" onclick=\"setMode('cal')\">&#128197; Cal</button>\n"
        "          <button id=\"btn-biz\"  class=\"sb-btn\"        onclick=\"setMode('biz')\">&#128188; Biz</button>\n"
        '        </div>\n'
        '      </div>\n'
        '      <div class="sb-section">\n'
        '        <span class="sb-lbl">Unit</span>\n'
        '        <div class="sb-toggle">\n'
        "          <button id=\"btn-days\"  class=\"sb-btn active\" onclick=\"setUnit('days')\">Days</button>\n"
        "          <button id=\"btn-hours\" class=\"sb-btn\"        onclick=\"setUnit('hours')\">Hours</button>\n"
        '        </div>\n'
        '      </div>\n'
        '      <hr class="sb-divider">\n'
        '      <div class="sb-section">\n'
        '        <span class="sb-lbl">Object</span>\n'
        '        <div class="sb-chips" id="chips-object">\n'
        '          <button class="chip on" data-v="all"      onclick="setObjectFilter(this)">All</button>\n'
        '          <button class="chip"    data-v="quote"    onclick="setObjectFilter(this)">Quote</button>\n'
        '          <button class="chip"    data-v="docusign" onclick="setObjectFilter(this)">DocuSign</button>\n'
        '          <button class="chip"    data-v="contract" onclick="setObjectFilter(this)">Contract</button>\n'
        '          <button class="chip"    data-v="order"    onclick="setObjectFilter(this)">Order</button>\n'
        '        </div>\n'
        '      </div>\n'
        '      <div class="sb-section">\n'
        '        <span class="sb-lbl">Month</span>\n'
        '        <div class="sb-chips" id="chips-month">\n'
        '          <button class="chip on" data-v="all" onclick="setFilter(\'month\',this)">All</button>\n'
        '        </div>\n'
        '      </div>\n'
        '      <div class="sb-section">\n'
        '        <span class="sb-lbl">Rework</span>\n'
        '        <div class="sb-chips" id="chips-rework">\n'
        '          <button class="chip on" data-v="all"    onclick="setFilter(\'rework\',this)">All</button>\n'
        '          <button class="chip"    data-v="rework" onclick="setFilter(\'rework\',this)">Rework</button>\n'
        '          <button class="chip"    data-v="clean"  onclick="setFilter(\'rework\',this)">Clean</button>\n'
        '        </div>\n'
        '      </div>\n'
        '      <div class="sb-section">\n'
        '        <span class="sb-lbl">NSCT</span>\n'
        '        <div class="sb-chips" id="chips-nsct">\n'
        '          <button class="chip on" data-v="all"      onclick="setFilter(\'nsct\',this)">All</button>\n'
        '          <button class="chip"    data-v="nsct"     onclick="setFilter(\'nsct\',this)">NSCT</button>\n'
        '          <button class="chip"    data-v="non_nsct" onclick="setFilter(\'nsct\',this)">Non-NSCT</button>\n'
        '        </div>\n'
        '      </div>\n'
        '      <div class="sb-section">\n'
        '        <span class="sb-lbl">Outcome</span>\n'
        '        <div class="sb-chips" id="chips-outcome">\n'
        '          <button class="chip on" data-v="all"         onclick="setFilter(\'outcome\',this)">All</button>\n'
        '          <button class="chip"    data-v="Accepted"    onclick="setFilter(\'outcome\',this)">Accepted</button>\n'
        '          <button class="chip"    data-v="Rejected"    onclick="setFilter(\'outcome\',this)">Rejected</button>\n'
        '          <button class="chip"    data-v="Denied"      onclick="setFilter(\'outcome\',this)">Denied</button>\n'
        '          <button class="chip"    data-v="In Progress" onclick="setFilter(\'outcome\',this)">In Progress</button>\n'
        '        </div>\n'
        '      </div>\n'
        '      <div class="sb-section">\n'
        '        <span class="sb-lbl">Resend-suspected</span>\n'
        '        <div class="sb-chips" id="chips-resend">\n'
        '          <button class="chip on" data-v="all"     onclick="setFilter(\'resend\',this)">All</button>\n'
        '          <button class="chip"    data-v="exclude" onclick="setFilter(\'resend\',this)">Exclude</button>\n'
        '        </div>\n'
        '      </div>\n'
        '    </div>\n'
        # ── Main content ──────────────────────────────────────────────────────
        '    <div class="main">\n'
        '      <div class="tabs">\n'
        "        <button id=\"tab-a\" class=\"tab active\" onclick=\"setView('A')\">&#128202; Velocity</button>\n"
        "        <button id=\"tab-b\" class=\"tab\"        onclick=\"setView('B')\">&#128269; Per-Quote Timeline</button>\n"
        "        <button id=\"tab-c\" class=\"tab\"        onclick=\"setView('C')\">&#128200; Statistical Analysis</button>\n"
        '      </div>\n'
        '      <div id="view-a">\n'
        '        <div id="callout-rework" class="callout hidden"></div>\n'
        '        <div class="dv-toggle">\n'
        "          <button id=\"dv-table\" class=\"dv-btn active\" onclick=\"setDataView('table')\">&#9638; Table</button>\n"
        "          <button id=\"dv-chart\" class=\"dv-btn\"        onclick=\"setDataView('chart')\">&#128202; Chart</button>\n"
        '        </div>\n'
        '        <div class="tbl-wrap" id="tbl-a"></div>\n'
        '        <div id="charts-a" class="hidden"></div>\n'
        '        <div id="summary-a"></div>\n'
        '      </div>\n'
        '      <div id="view-b" class="hidden">\n'
        '        <div class="search-box"><div class="search-row">\n'
        '          <input id="si" type="text" placeholder="Quote Number (e.g. Q12345) or Opportunity ID" onkeydown="if(event.key===\'Enter\')doSearch()">\n'
        '          <button onclick="doSearch()">Search</button>\n'
        '        </div></div>\n'
        '        <div id="sr"></div>\n'
        '      </div>\n'
        '      <div id="view-c" class="hidden">\n'
        '        <div id="callout-resend" class="callout hidden"></div>\n'
        '        <div class="tbl-wrap" id="stats-c"></div>\n'
        '        <div class="chart-card">\n'
        '          <div class="chart-hdr">Distribution by Step Pair</div>\n'
        '          <div class="chart-body">\n'
        '            <select id="stat-sel" class="stat-select" onchange="setStatPair(this.value)"></select>\n'
        '            <div id="stat-hist"></div>\n'
        '          </div>\n'
        '        </div>\n'
        '      </div>\n'
        '    </div>\n'
        '  </div>\n'
        '</div>\n'
    )

    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="UTF-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
        "<title>O2C Velocity Dashboard</title>\n"
        f"<style>{css}</style>\n"
        "</head>\n<body>\n"
        f"{body}"
        "<script>\n"
        f"const QUOTES={quotes_json};\n"
        f"{js}"
        "</script>\n"
        "</body>\n</html>"
    )
