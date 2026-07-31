"""
app.py

STAGE C: the website - Streamlit.

This file contains almost no NEW logic - it just calls the same
functions from pdf_text_extractor.py, extract_financials.py,
ratio_engine.py, and generate_insights.py, all of which are UI-framework
agnostic. The interface is a coat of paint on a working engine, not a
rebuild.

Two things changed from the very first version of this file:
1. Bank/financial-institution support: the original version predates
   ratio_engine.py's is_financial_institution() / calculate_bank_ratios(),
   so it never showed the Financial Institution Metrics section - which
   is why JPMorgan's numbers looked "half missing" the first time it was
   tested here even after the backend already supported banks.
2. A defensive guard around `years` being empty: the original crashed the
   whole app (Streamlit's "Oh no" screen) on any filing where extraction
   came back with zero usable years, because `st.columns(len(years))`
   raises when len(years) is 0. That's now caught with a clean error
   message instead.

To run this locally:
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=your_key_here
    streamlit run app.py
"""

import json
import tempfile
import os

import streamlit as st
import pandas as pd

from pdf_text_extractor import extract_clean_text
from extract_financials import build_financial_statements_excerpt, extract_financials_from_text
from ratio_engine import (
    calculate_ratios,
    calculate_bank_ratios,
    format_ratio,
    format_bank_ratio,
    is_financial_institution,
)
from generate_insights import generate_insights

st.set_page_config(page_title="AI Financial Health Analyzer", page_icon="📊", layout="wide")

# ---------------------------------------------------------------------------
# Look and feel - a light, card-based theme meant to read as "analyst
# tool," not "consumer app": a dark header band, off-white background,
# and Streamlit's own st.container(border=True) for the card look (no
# custom CSS needed for that part - it's a native Streamlit component).
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
    .stApp { background: #f4f5f7; }
    .hero {
        background: #0f172a;
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 24px;
    }
    .hero h1 { color: #ffffff; font-weight: 800; font-size: 28px; margin: 0 0 6px 0; }
    .hero p { color: #94a3b8; font-size: 15px; margin: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero">'
    '<h1>📊 AI Financial Health Analyzer</h1>'
    "<p>Upload a company's annual report (PDF) and get an automated financial "
    "health analysis: key figures, ratios, and AI-written commentary — all "
    "traceable back to the numbers actually found in the document.</p>"
    "</div>",
    unsafe_allow_html=True,
)


def _ratios_to_dataframe(ratios_by_year: dict, is_bank_by_year: dict) -> pd.DataFrame:
    """Rows = ratios, columns = years, using the same formatting rules as
    the CLI's format_ratio (percentages for margins/ROA/ROE, n/a labels
    that distinguish genuinely missing data from "not applicable to
    banks")."""
    years = sorted(ratios_by_year.keys())
    if not years:
        return pd.DataFrame()
    ratio_names = list(next(iter(ratios_by_year.values())).keys())
    rows = []
    for name in ratio_names:
        row = {"Ratio": name}
        for year in years:
            value = ratios_by_year[year].get(name)
            formatted = format_ratio(name, value, is_bank=is_bank_by_year.get(year, False)).strip()
            row[year] = formatted[len(name):].strip()
        rows.append(row)
    return pd.DataFrame(rows).set_index("Ratio")


def _bank_ratios_to_dataframe(bank_ratios_by_year: dict) -> pd.DataFrame:
    years = sorted(bank_ratios_by_year.keys())
    if not years:
        return pd.DataFrame()
    ratio_names = list(next(iter(bank_ratios_by_year.values())).keys())
    rows = []
    for name in ratio_names:
        row = {"Ratio": name}
        for year in years:
            value = bank_ratios_by_year[year].get(name)
            formatted = format_bank_ratio(name, value).strip()
            row[year] = formatted[len(name):].strip()
        rows.append(row)
    return pd.DataFrame(rows).set_index("Ratio")


def _headline_metrics(ratios_by_year, bank_ratios_by_year, is_bank_by_year, years):
    """4 st.metric() cards for the latest year, with a delta vs. the prior
    year where available. delta_color="inverse" is used for ratios where
    lower is generally better (debt/equity, efficiency ratio), so a
    decrease still shows green."""
    latest = years[-1]
    prior = years[-2] if len(years) > 1 else None

    if is_bank_by_year.get(latest):
        br_latest = bank_ratios_by_year.get(latest, {})
        br_prior = bank_ratios_by_year.get(prior, {}) if prior else {}
        specs = [
            ("Return on Equity", "return_on_equity", True, "normal"),
            ("Return on Assets", "return_on_assets", True, "normal"),
            ("Efficiency Ratio", "efficiency_ratio", True, "inverse"),
            ("Loans / Deposits", "loans_to_deposits_ratio", True, "off"),
        ]
        latest_vals, prior_vals = br_latest, br_prior
    else:
        r_latest = ratios_by_year.get(latest, {})
        r_prior = ratios_by_year.get(prior, {}) if prior else {}
        specs = [
            ("Net Profit Margin", "net_profit_margin", True, "normal"),
            ("Current Ratio", "current_ratio", False, "off"),
            ("Return on Equity", "roe", True, "normal"),
            ("Debt / Equity", "debt_to_equity", False, "inverse"),
        ]
        latest_vals, prior_vals = r_latest, r_prior

    cols = st.columns(4)
    for col, (label, key, is_pct, delta_color) in zip(cols, specs):
        value = latest_vals.get(key)
        if value is None:
            col.metric(f"{label} ({latest})", "n/a")
            continue
        display = f"{value*100:.1f}%" if is_pct else f"{value:.2f}"
        delta = None
        prior_value = prior_vals.get(key)
        if prior_value is not None:
            diff = value - prior_value
            delta = f"{diff*100:+.1f}pp" if is_pct else f"{diff:+.2f}"
        col.metric(f"{label} ({latest})", display, delta=delta, delta_color=delta_color)


def _trend_dataframe(ratios_by_year, bank_ratios_by_year, is_bank_by_year, years):
    """Net profit margin for ordinary companies, ROE for banks - picked
    because both are present across the widest range of filers."""
    rows = []
    for year in years:
        if is_bank_by_year.get(year):
            value = bank_ratios_by_year.get(year, {}).get("return_on_equity")
            metric = "Return on Equity (%)"
        else:
            value = ratios_by_year.get(year, {}).get("net_profit_margin")
            metric = "Net Profit Margin (%)"
        if value is not None:
            rows.append({"Year": year, metric: round(value * 100, 2)})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("Year")


with st.container(border=True):
    uploaded_file = st.file_uploader("Upload an annual report PDF", type=["pdf"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        with st.spinner("Reading the PDF..."):
            full_text = extract_clean_text(tmp_path)
    except Exception as e:
        st.error(f"Could not read the PDF: {e}")
        st.stop()
    finally:
        os.unlink(tmp_path)  # clean up the temporary file

    st.success(f"Read {len(full_text):,} characters from the report.")

    with st.spinner("Asking Claude to extract the financial figures..."):
        try:
            excerpt, missing_statements = build_financial_statements_excerpt(full_text)
            financials = extract_financials_from_text(excerpt)
        except Exception as e:
            st.error(f"Extraction failed: {e}")
            st.stop()

    if missing_statements:
        st.warning(
            f"Could not confidently find these statement(s): {', '.join(missing_statements)}. "
            f"Fields depending on them will show as missing below."
        )

    years = sorted(financials.get("years", {}).keys())
    if not years:
        st.error(
            "Couldn't extract any usable financial data from this document. "
            "This can happen with scanned/image-only PDFs, or a report format "
            "not yet covered by the statement-locating logic."
        )
        st.stop()

    st.success(f"Extracted data for: {', '.join(years)}")

    is_bank_by_year = {
        year: is_financial_institution(year_data)
        for year, year_data in financials["years"].items()
    }
    any_bank_year = any(is_bank_by_year.values())
    if any_bank_year:
        st.info(
            "🏦 This looks like a bank/financial institution filing. Current ratio, "
            "cash ratio, gross margin, operating margin, and free cash flow don't "
            "apply to how banks report (no classified balance sheet, no cost-of-"
            "revenue line) - see the Financial Institution Metrics table below instead."
        )

    ratios_by_year = {y: calculate_ratios(d) for y, d in financials["years"].items()}
    bank_ratios_by_year = {
        y: calculate_bank_ratios(d) for y, d in financials["years"].items() if is_bank_by_year.get(y)
    }

    st.subheader("Headline Metrics")
    _headline_metrics(ratios_by_year, bank_ratios_by_year, is_bank_by_year, years)

    trend_df = _trend_dataframe(ratios_by_year, bank_ratios_by_year, is_bank_by_year, years)
    if len(trend_df) > 1:
        with st.container(border=True):
            st.markdown("**Year-over-Year Trend**")
            st.line_chart(trend_df)

    with st.expander("See raw extracted figures"):
        st.json(financials)

    with st.container(border=True):
        st.subheader("Key Ratios by Year")
        st.dataframe(_ratios_to_dataframe(ratios_by_year, is_bank_by_year), width="stretch")

    if any_bank_year:
        with st.container(border=True):
            st.subheader("🏦 Financial Institution Metrics")
            st.caption(
                "Shown only for banks/financial institutions, using the company's own "
                "reported figures (from its Selected Financial Data / Financial Highlights "
                "table) rather than derived estimates."
            )
            st.dataframe(_bank_ratios_to_dataframe(bank_ratios_by_year), width="stretch")

    analysis_data = {
        "raw_financials": financials,
        "ratios": ratios_by_year,
        "bank_ratios": bank_ratios_by_year,
    }

    with st.spinner("Writing analyst commentary..."):
        try:
            narrative = generate_insights(analysis_data)
        except Exception as e:
            st.error(f"Commentary generation failed: {e}")
            st.stop()

    with st.container(border=True):
        st.subheader("AI-Written Analysis")
        st.markdown(narrative)

    st.download_button(
        "Download full analysis (JSON)",
        data=json.dumps(analysis_data, indent=2),
        file_name="analysis_output.json",
        mime="application/json",
    )
else:
    st.info("Upload a PDF above to get started.")
