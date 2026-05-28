"""
data_pipeline.py — O2C Velocity Analysis
Loads O2C_v1.xlsx (uploaded as bytes), joins all tabs, extracts 16 event
timestamps per quote, computes segmentation flags and step-pair deltas.

Ground truth: resources/velocity-analysis-reference.md §10 (column names,
join keys, step filter logic).  §2 / §3 of that doc describe the original
anonymized files and are NOT used here.
"""

import io
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
