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
    "TechReview",           # s02
    "TechApproved",         # s03
    "CommercialReview",     # s04
    "Commercial Approved",  # s05
    "NSCT Review",          # s06
    "Fully Approved",       # s07
    "Presented",            # s08
    "Accepted",             # s09
    "Signature Sent",       # s10
    "Customer Signed",      # s11
    "Fully Executed",       # s12
    "Contract Activated",   # s13
    "Order Activated",      # s14
    "Awaiting Install",     # s15
    "Deployment Closed",    # s16
]

PAIR_LABELS = [
    f"{STEP_LABELS[i]} → {STEP_LABELS[i + 1]}" for i in range(15)
]

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

    # Parse numeric contract ID from name pattern "… -C{NNNNN}"
    df["contract_number"] = (
        df["contract_name"]
        .str.extract(r"-C(\d+)\s*$", expand=False)
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
    """Steps 10–12 from DocuSign, joined on contract_number."""
    # Ensure join key types match
    qt["contract_number"] = qt["contract_number"].astype("Int64")

    s10 = (
        ds[ds["routing_order"] == 1]
        .groupby("contract_number", sort=False)["date_sent"].min()
        .rename("s10")
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
    for s in [s10, s11, s12]:
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

        # ── Phase spans: Quote(s01→s09), DocuSign(s09→s12), Contract(s12→s13),
        #                 Order(s13→s16), Full(s01→s16)  ─────────────────────
        _phase_ep = [(0, 8), (8, 11), (11, 12), (12, 15), (0, 15)]
        phase_cal_v: list = []
        phase_biz_v: list = []
        for s_idx, e_idx in _phase_ep:
            ts_s = row.get(f"s{s_idx + 1:02d}")
            ts_e = row.get(f"s{e_idx + 1:02d}")
            if pd.notna(ts_s) and pd.notna(ts_e):
                ts_s_t = pd.Timestamp(ts_s)
                ts_e_t = pd.Timestamp(ts_e)
                phase_cal_v.append((ts_e_t - ts_s_t).total_seconds() / 3600)
                phase_biz_v.append(float(np.busday_count(
                    np.datetime64(ts_s_t.date(), "D"),
                    np.datetime64(ts_e_t.date(), "D"),
                    holidays=US_HOLIDAY_DATES,
                )))
            else:
                phase_cal_v.append(None)
                phase_biz_v.append(None)

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
"""

    js = r"""
const STEP_LABELS=["Quote Created","TechReview","TechApproved","CommercialReview","Commercial Approved","NSCT Review","Fully Approved","Presented","Accepted","Signature Sent to Customer","Customer Signed","Fully Executed","Contract Activated","Order Activated","Awaiting Install","Deployment Closed"];
const PAIR_LABELS=["Quote Created → TechReview","TechReview → TechApproved","TechApproved → CommercialReview","CommercialReview → Commercial Approved","Commercial Approved → NSCT Review","NSCT Review → Fully Approved","Fully Approved → Presented","Presented → Accepted","Accepted → Signature Sent to Customer","Signature Sent to Customer → Customer Signed","Customer Signed → Fully Executed","Fully Executed → Contract Activated","Contract Activated → Order Activated","Order Activated → Awaiting Install","Awaiting Install → Deployment Closed"];
const OBJECT_PAIRS={all:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],quote:[0,1,2,3,4,5,6,7],docusign:[8,9,10],contract:[11],order:[12,13,14]};
const PHASE_LABELS=['Quote Phase (Created → Accepted)','DocuSign Phase (Accepted → Fully Executed)','Contract Phase (Executed → Contract Activated)','Order Phase (Contract → Deployed)','Full Cycle (Created → Deployed)'];
/* object color palettes — indexed by pair (0-14) and step (0-15) */
const OBJ_COLORS=['#3b82f6','#3b82f6','#3b82f6','#3b82f6','#3b82f6','#3b82f6','#3b82f6','#3b82f6','#7c3aed','#7c3aed','#7c3aed','#0891b2','#059669','#059669','#059669'];
const STEP_COLORS=['#3b82f6','#3b82f6','#3b82f6','#3b82f6','#3b82f6','#3b82f6','#3b82f6','#3b82f6','#3b82f6','#7c3aed','#7c3aed','#7c3aed','#0891b2','#059669','#059669','#059669'];
/* short destination-step label for each of the 15 pairs (shown below heatmap) */
const DEST_LABELS=['TechReview','TechApproved','Comm. Review','Comm. Approved','NSCT Review','Fully Approved','Presented','Accepted','Sig. Sent','Cust. Signed','Fully Exec.','Cntr. Activated','Order Active','Awaiting Install','Deployed'];

const state={mode:'cal',unit:'days',rework:'all',nsct:'all',outcome:'all',month:'all',object:'all'};

function avg(arr){const v=arr.filter(x=>x!==null&&!isNaN(x));return v.length?v.reduce((a,b)=>a+b,0)/v.length:null;}
function med(arr){const v=[...arr.filter(x=>x!==null&&!isNaN(x))].sort((a,b)=>a-b);if(!v.length)return null;const m=Math.floor(v.length/2);return v.length%2?v[m]:(v[m-1]+v[m])/2;}
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
  let html='<div class="summary"><div class="summary-hdr">End-to-End Summary — '+un+'</div>'
    +'<table class="vtbl"><thead><tr><th>Phase</th><th class="r">Avg ('+un+')</th>'
    +'<th class="r">Median</th><th class="r">n</th></tr></thead><tbody>';
  PHASE_LABELS.forEach(function(lbl,i){
    const vals=filtered.map(function(q){return getPhaseVal(q,i);}).filter(function(x){return x!==null;});
    const a=avg(vals),md=med(vals),n=vals.length;
    const sep=i===4?' style="border-top:2px solid #e2e8f0;font-weight:600"':'';
    html+='<tr'+sep+'>'
      +'<td class="pair">'+escHtml(lbl)+'</td>'
      +'<td class="'+(a===null?'muted':'num')+'">'+fmt(a)+'</td>'
      +'<td class="'+(md===null?'muted':'num')+'">'+fmt(md)+'</td>'
      +'<td class="num">'+n+'</td></tr>';
  });
  html+='</tbody></table></div>';
  document.getElementById('summary-a').innerHTML=html;
}

/* ── Chart rendering ─────────────────────────────────────────────────────── */
function renderBarChart(allRows,u){
  const LW=210,BW=440,VW=70,TW=LW+BW+VW;
  const rowH=26,gap=3,padT=14,padB=10;
  const H=padT+15*(rowH+gap)+padB;
  const avgs=allRows.map(r=>r.a);
  const maxV=Math.max(...avgs.filter(x=>x!==null),0.001);
  let s=`<svg viewBox="0 0 ${TW} ${H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;display:block" font-family="system-ui,sans-serif">`;
  allRows.forEach((r,i)=>{
    const y=padT+i*(rowH+gap);
    const bw=r.a!==null?(r.a/maxV)*BW:0;
    const col=OBJ_COLORS[r.pairIdx];
    const lbl=r.lbl.length>30?r.lbl.substring(0,28)+'…':r.lbl;
    const warn=r.pairIdx===14?' ⚠':'';
    s+=`<rect x="0" y="${y}" width="${TW}" height="${rowH}" fill="${i%2===0?'#f8fafc':'#fff'}"/>`;
    s+=`<text x="${LW-8}" y="${y+rowH/2+4}" text-anchor="end" font-size="11" fill="#374151">${escHtml(lbl)}${warn}</text>`;
    if(bw>0)s+=`<rect x="${LW}" y="${y+5}" width="${bw}" height="${rowH-10}" fill="${col}" rx="3" opacity="0.85"/>`;
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
  const W=860,padL=20,padR=20,barW=W-padL-padR;
  /* barY pushed down to leave room for above-bar labels */
  const barY=72,barH=52,H=240;
  const avgs=allRows.map(r=>r.a!==null?r.a:0);
  const total=avgs.reduce((a,b)=>a+b,0)||1;
  const nonNull=allRows.map(r=>r.a).filter(x=>x!==null);
  const minV=nonNull.length?Math.min(...nonNull):0;
  const maxV=nonNull.length?Math.max(...nonNull):1;
  const range=maxV-minV||1;

  /* cumulative x per segment boundary */
  const xs=[padL];
  avgs.forEach(v=>xs.push(xs[xs.length-1]+(v/total)*barW));

  const phases=[
    {s:0,e:8,col:'#3b82f6',lbl:'Quote (1–8)'},
    {s:8,e:11,col:'#7c3aed',lbl:'DocuSign (9–11)'},
    {s:11,e:12,col:'#0891b2',lbl:'Contract (12)'},
    {s:12,e:15,col:'#059669',lbl:'Order (13–15)'}
  ];

  let s=`<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;display:block" font-family="system-ui,sans-serif">`;

  /* ── alternating labels: even → above bar, odd → below bar ── */
  allRows.forEach((r,i)=>{
    const cx=(xs[i]+xs[i+1])/2;
    const lbl=DEST_LABELS[i]+(i===14?' ⚠':'');
    const isAbove=i%2===0;
    if(isAbove){
      /* tick up from bar top */
      s+=`<line x1="${cx}" y1="${barY-2}" x2="${cx}" y2="${barY-10}" stroke="#cbd5e1" stroke-width="1"/>`;
      /* label rotated -45°, anchored below the pivot so it reads up-left */
      s+=`<text transform="translate(${cx},${barY-12}) rotate(-45)" text-anchor="start" font-size="10" fill="#374151" font-weight="500">${escHtml(lbl)}</text>`;
    } else {
      /* tick down from bar bottom */
      s+=`<line x1="${cx}" y1="${barY+barH+2}" x2="${cx}" y2="${barY+barH+10}" stroke="#cbd5e1" stroke-width="1"/>`;
      /* label rotated -45°, anchored so it reads down-left */
      s+=`<text transform="translate(${cx},${barY+barH+12}) rotate(-45)" text-anchor="end" font-size="10" fill="#374151" font-weight="500">${escHtml(lbl)}</text>`;
    }
  });

  /* ── segments ── */
  allRows.forEach((r,i)=>{
    const x1=xs[i],x2=xs[i+1],sw=x2-x1;
    const t=r.a!==null?(r.a-minV)/range:0;
    const fill=r.a!==null?heatColor(t):'#f1f5f9';
    const tc=r.a!==null?heatTextColor(t):'#94a3b8';
    /* fill */
    s+=`<rect x="${x1}" y="${barY}" width="${sw}" height="${barH}" fill="${fill}"/>`;
    /* phase-colour top strip (4px) */
    s+=`<rect x="${x1}" y="${barY}" width="${sw}" height="4" fill="${OBJ_COLORS[i]}" opacity="0.6"/>`;
    /* white divider */
    if(i>0)s+=`<line x1="${x1}" y1="${barY}" x2="${x1}" y2="${barY+barH}" stroke="#fff" stroke-width="1.5"/>`;
    /* step number — larger font for readability */
    const fs=sw>28?14:sw>16?11:9;
    const numY=barY+barH/2+(sw>30?2:6);
    s+=`<text x="${(x1+x2)/2}" y="${numY}" text-anchor="middle" font-size="${fs}" font-weight="700" fill="${tc}">${i+1}</text>`;
    /* value label inside segment only if wide enough */
    if(sw>42)s+=`<text x="${(x1+x2)/2}" y="${barY+barH/2+16}" text-anchor="middle" font-size="9.5" fill="${tc}" opacity="0.9">${r.a!==null?fmt(r.a)+u:''}</text>`;
  });

  /* bar outline */
  s+=`<rect x="${padL}" y="${barY}" width="${barW}" height="${barH}" fill="none" stroke="#cbd5e1" stroke-width="1"/>`;

  /* ── bottom legend row ── */
  const lgY=H-14;
  /* gradient swatch */
  const lgW=90,lgH=7,swatchSteps=18;
  for(let i=0;i<swatchSteps;i++){
    s+=`<rect x="${padL+i*(lgW/swatchSteps)}" y="${lgY}" width="${lgW/swatchSteps+0.5}" height="${lgH}" fill="${heatColor(i/(swatchSteps-1))}"/>`;
  }
  s+=`<rect x="${padL}" y="${lgY}" width="${lgW}" height="${lgH}" fill="none" stroke="#e2e8f0" stroke-width="0.5"/>`;
  s+=`<text x="${padL}" y="${lgY-3}" font-size="8.5" fill="#94a3b8">Less time</text>`;
  s+=`<text x="${padL+lgW}" y="${lgY-3}" text-anchor="end" font-size="8.5" fill="#94a3b8">More time</text>`;
  /* phase legend (inline, right of gradient) */
  let plx=padL+lgW+24;
  phases.forEach(ph=>{
    s+=`<rect x="${plx}" y="${lgY}" width="${lgH+1}" height="${lgH}" fill="${ph.col}" rx="1"/>`;
    s+=`<text x="${plx+lgH+4}" y="${lgY+lgH-0.5}" font-size="9.5" fill="#374151">${ph.lbl}</text>`;
    plx+=140;
  });
  /* total label (far right) */
  s+=`<text x="${padL+barW}" y="${lgY+lgH}" text-anchor="end" font-size="9.5" fill="#94a3b8">Total avg: ${total>0?fmt(total)+u:'—'}</text>`;

  s+='</svg>';
  return s;
}

function renderCharts(allRows,u){
  document.getElementById('charts-a').innerHTML=
    '<div class="chart-card"><div class="chart-hdr">Avg Time per Step Pair — '+unitName()+'</div>'
    +'<div class="chart-body">'+renderBarChart(allRows,u)+'</div></div>'
    +'<div class="chart-card"><div class="chart-hdr">Where Time Is Spent — Proportional Heatmap</div>'
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
    return true;
  });
}

function setMode(m){
  state.mode=m;
  document.getElementById('btn-cal').classList.toggle('active',m==='cal');
  document.getElementById('btn-biz').classList.toggle('active',m==='biz');
  renderViewA();const sv=document.getElementById('si').value.trim();if(sv)renderViewB(sv);
}
function setUnit(u){
  state.unit=u;
  document.getElementById('btn-days').classList.toggle('active',u==='days');
  document.getElementById('btn-hours').classList.toggle('active',u==='hours');
  renderViewA();const sv=document.getElementById('si').value.trim();if(sv)renderViewB(sv);
}
function setView(v){
  document.getElementById('view-a').classList.toggle('hidden',v!=='A');
  document.getElementById('view-b').classList.toggle('hidden',v!=='B');
  document.getElementById('tab-a').classList.toggle('active',v==='A');
  document.getElementById('tab-b').classList.toggle('active',v==='B');
}
function setFilter(key,btn){
  const grp=btn.closest('.sb-chips,.chips');
  grp.querySelectorAll('.chip').forEach(c=>c.classList.remove('on'));
  btn.classList.add('on');state[key]=btn.dataset.v;renderViewA();
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
})();
renderViewA();
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
        '    </div>\n'
        # ── Main content ──────────────────────────────────────────────────────
        '    <div class="main">\n'
        '      <div class="tabs">\n'
        "        <button id=\"tab-a\" class=\"tab active\" onclick=\"setView('A')\">&#128202; Aggregate Velocity</button>\n"
        "        <button id=\"tab-b\" class=\"tab\"        onclick=\"setView('B')\">&#128269; Per-Quote Timeline</button>\n"
        '      </div>\n'
        '      <div id="view-a">\n'
        '        <div id="callout-rework" class="callout hidden"></div>\n'
        '        <div class="tbl-wrap" id="tbl-a"></div>\n'
        '        <div id="charts-a"></div>\n'
        '        <div id="summary-a"></div>\n'
        '      </div>\n'
        '      <div id="view-b" class="hidden">\n'
        '        <div class="search-box"><div class="search-row">\n'
        '          <input id="si" type="text" placeholder="Quote Number (e.g. Q12345) or Opportunity ID" onkeydown="if(event.key===\'Enter\')doSearch()">\n'
        '          <button onclick="doSearch()">Search</button>\n'
        '        </div></div>\n'
        '        <div id="sr"></div>\n'
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
