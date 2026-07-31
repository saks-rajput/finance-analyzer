"""
app.py

STAGE C: the website - now built with Gradio instead of Streamlit.

Why the switch: the Streamlit Community Cloud deployment kept hitting
flaky infra issues (module-cache KeyErrors during redeploys, the app
catching mid-git-push states). None of that was a bug in this app's own
code - it was Streamlit Cloud's hosting behavior. Gradio apps deploy
cleanly on platforms like Render or Google Cloud Run, which build a
fresh container on every deploy rather than hot-reloading a live
process, and is a very common home for exactly this kind of "upload a
file, run some AI, show results" app.

This file contains almost no NEW logic - it just calls the same
functions from pdf_text_extractor.py, extract_financials.py,
ratio_engine.py, and generate_insights.py that were already validated
under the Streamlit version. The interface is a new coat of paint on a
working engine, not a rebuild.

To run this locally:
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=your_key_here
    python app.py
Then open the local URL Gradio prints (usually http://127.0.0.1:7860).

See README.md for deploying this on Render or Google Cloud Run.
"""

import json
import tempfile
import os

import gradio as gr
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

# ---------------------------------------------------------------------------
# Look and feel
# ---------------------------------------------------------------------------
# A light, card-based theme meant to read as "analyst tool," not "consumer
# app" - dark slate header band, off-white page background, white cards
# with a soft shadow, one restrained accent color used only for numbers
# and highlights so the actual data stays the visual focus.
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background: #f4f5f7 !important;
    max-width: 1080px !important;
    margin: 0 auto !important;
}

#hero {
    background: #0f172a;
    border-radius: 16px;
    padding: 32px 36px;
    margin-bottom: 20px;
}
#hero h1 {
    color: #ffffff !important;
    font-weight: 800 !important;
    font-size: 28px !important;
    margin-bottom: 6px !important;
}
#hero p {
    color: #94a3b8 !important;
    font-size: 15px !important;
    margin: 0 !important;
}

.card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 20px 22px;
    margin-bottom: 18px;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}

.section-title {
    font-weight: 700 !important;
    font-size: 17px !important;
    color: #0f172a !important;
    margin-bottom: 4px !important;
}
.section-subtitle {
    color: #6b7280 !important;
    font-size: 13px !important;
}

.stat-row {
    display: flex;
    gap: 14px;
    flex-wrap: wrap;
    margin-bottom: 18px;
}
.stat-card {
    flex: 1 1 200px;
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 18px 20px;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}
.stat-card .stat-label {
    color: #6b7280;
    font-size: 12.5px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 6px;
}
.stat-card .stat-value {
    color: #0f172a;
    font-size: 28px;
    font-weight: 800;
    line-height: 1.1;
}
.stat-card .stat-value.accent { color: #b45309; }
.stat-card .stat-value.muted { color: #9ca3af; font-size: 20px; }

#status_box textarea {
    background: #f0fdf4 !important;
    border-color: #bbf7d0 !important;
    color: #14532d !important;
    font-size: 13.5px !important;
}
"""


def _fmt_stat(value, is_pct):
    if value is None:
        return '<span class="stat-value muted">n/a</span>'
    text = f"{value*100:.1f}%" if is_pct else f"{value:.2f}"
    return f'<span class="stat-value accent">{text}</span>'


def _build_headline_stats_html(ratios_by_year, bank_ratios_by_year, is_bank_by_year):
    """A row of 4 big-number stat cards for the most recent year - the
    kind of "glanceable" summary a reviewer skims before reading further."""
    years = sorted(set(ratios_by_year) | set(bank_ratios_by_year))
    if not years:
        return ""
    latest = years[-1]

    if is_bank_by_year.get(latest):
        br = bank_ratios_by_year.get(latest, {})
        cards = [
            ("Return on Equity", br.get("return_on_equity"), True),
            ("Return on Assets", br.get("return_on_assets"), True),
            ("Efficiency Ratio", br.get("efficiency_ratio"), True),
            ("Loans / Deposits", br.get("loans_to_deposits_ratio"), True),
        ]
    else:
        r = ratios_by_year.get(latest, {})
        cards = [
            ("Net Profit Margin", r.get("net_profit_margin"), True),
            ("Current Ratio", r.get("current_ratio"), False),
            ("Return on Equity", r.get("roe"), True),
            ("Debt / Equity", r.get("debt_to_equity"), False),
        ]

    cards_html = "".join(
        f'<div class="stat-card"><div class="stat-label">{label} ({latest})</div>{_fmt_stat(value, is_pct)}</div>'
        for label, value, is_pct in cards
    )
    return f'<div class="stat-row">{cards_html}</div>'


def _build_trend_dataframe(ratios_by_year, bank_ratios_by_year, is_bank_by_year):
    """Long-format dataframe for a year-over-year trend chart. Uses net
    profit margin for ordinary companies, return on equity for banks -
    picked because both are present across the widest range of filers."""
    years = sorted(set(ratios_by_year) | set(bank_ratios_by_year))
    rows = []
    for year in years:
        if is_bank_by_year.get(year):
            value = bank_ratios_by_year.get(year, {}).get("return_on_equity")
            metric = "Return on Equity"
        else:
            value = ratios_by_year.get(year, {}).get("net_profit_margin")
            metric = "Net Profit Margin"
        if value is not None:
            rows.append({"Year": year, "Value (%)": round(value * 100, 2), "Metric": metric})
    return pd.DataFrame(rows, columns=["Year", "Value (%)", "Metric"])


def _ratios_to_dataframe(ratios_by_year: dict, is_bank_by_year: dict) -> pd.DataFrame:
    """Turn {year: {ratio_name: value}} into a rows=ratios, columns=years
    table, using the exact same formatting rules as the CLI's format_ratio
    (percentages for margins/ROA/ROE, 2 decimals otherwise, n/a for None -
    or "not applicable to banks" for the handful of ratios that structurally
    don't exist for a financial institution's statements)."""
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
            # format_ratio returns "name    value" - keep just the value part
            row[year] = formatted[len(name):].strip()
        rows.append(row)
    return pd.DataFrame(rows)


def _bank_ratios_to_dataframe(bank_ratios_by_year: dict) -> pd.DataFrame:
    """Same idea as _ratios_to_dataframe, for the bank-specific metric set.
    Only years actually flagged as a financial institution are included."""
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
    return pd.DataFrame(rows)


def analyze_report(pdf_file):
    if pdf_file is None:
        raise gr.Error("Upload a PDF first.")

    pdf_path = pdf_file if isinstance(pdf_file, str) else pdf_file.name

    try:
        full_text = extract_clean_text(pdf_path)
    except Exception as e:
        raise gr.Error(f"Could not read the PDF: {e}")

    status_lines = [f"Read {len(full_text):,} characters from the report."]

    try:
        excerpt, missing_statements = build_financial_statements_excerpt(full_text)
    except Exception as e:
        raise gr.Error(f"Could not locate the financial statements: {e}")

    if missing_statements:
        status_lines.append(
            "⚠️ Could not confidently find: " + ", ".join(missing_statements) +
            ". Fields depending on these will show as missing below."
        )

    try:
        financials = extract_financials_from_text(excerpt)
    except Exception as e:
        raise gr.Error(f"Extraction failed: {e}")

    years = sorted(financials.get("years", {}).keys())
    status_lines.append(f"Extracted data for: {', '.join(years)}")

    is_bank_by_year = {
        year: is_financial_institution(year_data)
        for year, year_data in financials.get("years", {}).items()
    }
    any_bank_year = any(is_bank_by_year.values())
    if any_bank_year:
        status_lines.append(
            "🏦 This looks like a bank/financial institution filing. Current ratio, "
            "cash ratio, gross margin, operating margin, and free cash flow don't "
            "apply to how banks report (no classified balance sheet, no cost-of-"
            "revenue line) - see the Financial Institution Metrics table below instead."
        )

    ratios_by_year = {
        year: calculate_ratios(year_data)
        for year, year_data in financials.get("years", {}).items()
    }
    bank_ratios_by_year = {
        year: calculate_bank_ratios(year_data)
        for year, year_data in financials.get("years", {}).items()
        if is_bank_by_year.get(year)
    }

    analysis_data = {
        "raw_financials": financials,
        "ratios": ratios_by_year,
        "bank_ratios": bank_ratios_by_year,
    }

    try:
        narrative = generate_insights(analysis_data)
    except Exception as e:
        raise gr.Error(f"Commentary generation failed: {e}")

    stats_html = _build_headline_stats_html(ratios_by_year, bank_ratios_by_year, is_bank_by_year)
    trend_df = _build_trend_dataframe(ratios_by_year, bank_ratios_by_year, is_bank_by_year)
    ratios_df = _ratios_to_dataframe(ratios_by_year, is_bank_by_year)
    bank_ratios_df = _bank_ratios_to_dataframe(bank_ratios_by_year)

    # Write the downloadable JSON to a temp file, since Gradio's File
    # output component wants a path.
    tmp_json = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, prefix="analysis_output_"
    )
    json.dump(analysis_data, tmp_json, indent=2)
    tmp_json.close()

    return (
        "\n".join(status_lines),
        stats_html,
        financials,
        gr.update(value=trend_df, visible=len(trend_df) > 1),
        ratios_df,
        gr.update(visible=any_bank_year, value=bank_ratios_df),
        narrative,
        tmp_json.name,
    )


with gr.Blocks(title="AI Financial Health Analyzer", css=CUSTOM_CSS) as demo:
    gr.HTML(
        '<div id="hero">'
        '<h1>📊 AI Financial Health Analyzer</h1>'
        '<p>Upload a company\'s annual report (PDF) and get an automated financial '
        'health analysis: key figures, ratios, and AI-written commentary — all '
        'traceable back to the numbers actually found in the document.</p>'
        '</div>'
    )

    with gr.Group(elem_classes="card"):
        pdf_input = gr.File(label="Upload an annual report PDF", file_types=[".pdf"])
        analyze_btn = gr.Button("Analyze", variant="primary")

    status_box = gr.Textbox(label="Status", interactive=False, elem_id="status_box")

    stats_output = gr.HTML()

    with gr.Group(elem_classes="card"):
        gr.Markdown("Year-over-Year Trend", elem_classes="section-title")
        trend_plot = gr.LinePlot(
            x="Year", y="Value (%)", color="Metric",
            height=220, visible=False, show_label=False,
        )

    with gr.Accordion("See raw extracted figures", open=False):
        raw_json_output = gr.JSON()

    with gr.Group(elem_classes="card"):
        gr.Markdown("Key Ratios by Year", elem_classes="section-title")
        ratios_output = gr.Dataframe(interactive=False)

    with gr.Group(elem_classes="card"):
        gr.Markdown("🏦 Financial Institution Metrics", elem_classes="section-title")
        gr.Markdown(
            "Shown only for banks/financial institutions, using the company's own "
            "reported figures rather than derived estimates.",
            elem_classes="section-subtitle",
        )
        bank_ratios_output = gr.Dataframe(interactive=False, visible=False)

    with gr.Group(elem_classes="card"):
        gr.Markdown("AI-Written Analysis", elem_classes="section-title")
        narrative_output = gr.Markdown()

    download_output = gr.File(label="Download full analysis (JSON)")

    analyze_btn.click(
        fn=analyze_report,
        inputs=[pdf_input],
        outputs=[
            status_box,
            stats_output,
            raw_json_output,
            trend_plot,
            ratios_output,
            bank_ratios_output,
            narrative_output,
            download_output,
        ],
    )

if __name__ == "__main__":
    # Render, Cloud Run (and most non-Hugging-Face hosts) assign a port via
    # the $PORT env var and expect the app to bind to 0.0.0.0, not Gradio's
    # local default of 127.0.0.1:7860. Falling back to 7860 keeps
    # `python app.py` working unchanged for local testing.
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
