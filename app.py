"""
app.py

STAGE C: the website. This file turns everything you already built and
tested into an actual webpage with an upload box, using Streamlit.

Notice this file contains almost no NEW logic - it just calls the same
functions from pdf_text_extractor.py, extract_financials.py,
ratio_engine.py, and generate_insights.py that you already validated.
The website is a new coat of paint on a working engine, not a rebuild.

To run this on your own computer (to preview before putting it online):
    pip install streamlit
    streamlit run app.py
"""

import streamlit as st
import json
import tempfile
import os

from pdf_text_extractor import extract_clean_text
from extract_financials import build_financial_statements_excerpt, extract_financials_from_text
from ratio_engine import calculate_ratios, format_ratio
from generate_insights import generate_insights

st.set_page_config(page_title="AI Financial Health Analyzer", page_icon="📊", layout="wide")

st.title("📊 AI Financial Health Analyzer")
st.write(
    "Upload a company's annual report (PDF) and get an automated financial "
    "health analysis: key figures, ratios, and AI-written commentary — all "
    "traceable back to the numbers actually found in the document."
)

uploaded_file = st.file_uploader("Upload an annual report PDF", type=["pdf"])

if uploaded_file is not None:
    # Streamlit gives us the uploaded file in memory; our extractor expects
    # a file path on disk, so we save it to a temporary file first.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("Reading the PDF..."):
        full_text = extract_clean_text(tmp_path)
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
            f"Could not find these statement(s) in this document, using any known "
            f"heading wording: {', '.join(missing_statements)}. Fields that depend "
            f"on them will show as missing below."
        )
        with st.expander("Debug: show possible statement headings found in this document"):
            import re
            matches = list(re.finditer(r".{0,40}statements?.{0,60}", full_text, re.IGNORECASE))
            if matches:
                st.write(f"Found {len(matches)} lines mentioning 'statement' - here are the first 15:")
                for m in matches[:15]:
                    st.code(m.group().strip())
            else:
                st.write("No lines containing the word 'statement' were found at all in the extracted text.")
            st.write("Copy whichever line above looks like the real heading for the "
                     "missing statement(s) above, and send it over so the matching "
                     "list can be updated.")

    years = sorted(financials.get("years", {}).keys())
    st.success(f"Extracted data for: {', '.join(years)}")

    # Show the raw extracted figures so the user can eyeball them for sanity
    with st.expander("See raw extracted figures"):
        st.json(financials)

    # Calculate ratios - plain code, no AI, exactly as before
    ratios_by_year = {y: calculate_ratios(d) for y, d in financials["years"].items()}

    st.subheader("Key Ratios by Year")
    cols = st.columns(len(years))
    for col, year in zip(cols, years):
        with col:
            st.markdown(f"**{year}**")
            for name, value in ratios_by_year[year].items():
                st.text(format_ratio(name, value).strip())

    analysis_data = {"raw_financials": financials, "ratios": ratios_by_year}

    with st.spinner("Writing analyst commentary..."):
        try:
            narrative = generate_insights(analysis_data)
        except Exception as e:
            st.error(f"Commentary generation failed: {e}")
            st.stop()

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
