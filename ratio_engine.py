"""
ratio_engine.py

STEP 3 of the pipeline: turn raw numbers into the ratios analysts actually
use, e.g. current ratio, profit margins, debt-to-equity.

Notice there is NO AI anywhere in this file. This is 100% plain arithmetic,
the exact same formulas from your manual analysis. This is deliberate:
once the numbers are extracted and verified (Step 2), the math itself
should never be left to an AI to "calculate" - it should be code that
does the exact same division every single time, with zero chance of a
hallucinated number.

Usage:
    python ratio_engine.py result.json
"""

import sys
import json


def safe_div(numerator, denominator):
    """Divide safely - if a number is missing, return None instead of crashing."""
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def calculate_ratios(year_data: dict) -> dict:
    ca = year_data.get("total_current_assets")
    cl = year_data.get("total_current_liabilities")
    cash = year_data.get("cash_and_equivalents")
    ta = year_data.get("total_assets")
    tl = year_data.get("total_liabilities")
    revenue = year_data.get("revenue")
    cogs = year_data.get("cost_of_revenue")
    op_income = year_data.get("operating_income")
    net_income = year_data.get("net_income")
    ocf = year_data.get("operating_cash_flow")
    capex = year_data.get("capital_expenditures")  # expected to be negative

    gross_profit = None
    if revenue is not None and cogs is not None:
        gross_profit = revenue - cogs

    equity = None
    if ta is not None and tl is not None:
        equity = ta - tl

    free_cash_flow = None
    if ocf is not None and capex is not None:
        free_cash_flow = ocf + capex  # capex is already negative, so we add it

    return {
        "current_ratio": safe_div(ca, cl),
        "cash_ratio": safe_div(cash, cl),
        "debt_to_equity": safe_div(tl, equity),
        "debt_to_assets": safe_div(tl, ta),
        "gross_margin": safe_div(gross_profit, revenue),
        "operating_margin": safe_div(op_income, revenue),
        "net_profit_margin": safe_div(net_income, revenue),
        "roa": safe_div(net_income, ta),
        "roe": safe_div(net_income, equity),
        "free_cash_flow": free_cash_flow,
        "operating_cash_flow_margin": safe_div(ocf, revenue),
        "cash_conversion_ratio": safe_div(ocf, net_income),
    }


def format_ratio(name: str, value) -> str:
    if value is None:
        return f"  {name:28s} n/a (missing data)"
    if "margin" in name or "roa" in name or "roe" in name:
        return f"  {name:28s} {value*100:.1f}%"
    if name == "free_cash_flow":
        return f"  {name:28s} {value:,.0f}"
    return f"  {name:28s} {value:.2f}"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ratio_engine.py <path_to_result.json>", file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[1]) as f:
        data = json.load(f)

    for year, year_data in sorted(data.get("years", {}).items()):
        print(f"\n=== {year} ===")
        ratios = calculate_ratios(year_data)
        for name, value in ratios.items():
            print(format_ratio(name, value))
