"""
analyze_report.py

STAGE A: one script, one command, instead of four.

This ties together everything built so far:
  1. pdf_text_extractor.py  -> clean text
  2. extract_financials.py -> structured JSON (the one AI-powered step)
  3. ratio_engine.py       -> calculated ratios

Note on validation: test_extraction.py is deliberately NOT part of this
combined script. That test only knows the "correct answers" for the
Alphabet 2025 report we used to build and prove out this pipeline - it
can't validate a different company's numbers, since we have no manual
ground truth for them. Validation against a manual analysis is something
you do once per new source document while you still trust-but-verify;
it isn't part of the repeatable, everyday pipeline.

Usage:
    python analyze_report.py path/to/annual_report.pdf
"""

import sys
import json

from pdf_text_extractor import extract_clean_text
from extract_financials import build_financial_statements_excerpt, extract_financials_from_text
from ratio_engine import calculate_ratios, format_ratio


def run_pipeline(pdf_path: str) -> dict:
    print(f"[1/3] Reading and cleaning text from {pdf_path} ...")
    full_text = extract_clean_text(pdf_path)
    print(f"      Got {len(full_text):,} characters of text.")

    print("[2/3] Locating financial statements and asking Claude to extract figures ...")
    excerpt, missing_statements = build_financial_statements_excerpt(full_text)
    if missing_statements:
        print(f"      WARNING: could not find in this document: {', '.join(missing_statements)}")
        print("      Fields depending on these will show as missing/None below.")
    financials = extract_financials_from_text(excerpt)
    years_found = list(financials.get("years", {}).keys())
    print(f"      Extracted data for years: {', '.join(years_found)}")

    print("[3/3] Calculating ratios ...")
    ratios_by_year = {}
    for year, year_data in financials.get("years", {}).items():
        ratios_by_year[year] = calculate_ratios(year_data)

    return {"raw_financials": financials, "ratios": ratios_by_year}


def print_report(result: dict):
    print("\n" + "=" * 50)
    print("FINANCIAL HEALTH SNAPSHOT")
    print("=" * 50)
    for year in sorted(result["ratios"].keys()):
        print(f"\n--- {year} ---")
        for name, value in result["ratios"][year].items():
            print(format_ratio(name, value))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_report.py <path_to_pdf>", file=sys.stderr)
        sys.exit(1)

    result = run_pipeline(sys.argv[1])

    # Save the full detail to a file, and print a readable summary to the screen
    with open("analysis_output.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nFull detail saved to analysis_output.json")

    print_report(result)
