"""
extract_financials.py

STEP 2 of the pipeline: turn clean text into structured numbers.

This is the one piece that actually uses AI. Everything before it
(pdf_text_extractor.py) and everything after it (a future ratio_engine.py)
is plain, deterministic code - no AI, fully traceable. This is deliberate:
we only want AI doing the part it's genuinely good at (reading messy
prose/tables and mapping them to a schema), never the arithmetic - and
never the job of deciding "is this chunk of text actually the real
financial statement." That decision needs to be reliable and traceable
too, so it stays rule-based (see find_real_statement below).

Requires:
    pip install anthropic
    export ANTHROPIC_API_KEY=your_key_here   (get one at console.anthropic.com)

Usage:
    python extract_financials.py clean_text.txt
"""

import sys
import re
import json
import anthropic

# The schema is the contract between the AI and the rest of your program.
# Being explicit and strict here is what makes the output trustworthy.
SCHEMA_INSTRUCTIONS = """
You are a financial data extraction engine. You will be given raw text
from a company's annual report (income statement, balance sheet, and
cash flow statement sections may all be present, possibly with some
surrounding narrative text mixed in).

Extract the following line items for EVERY year you can find in the text.
Respond with ONLY valid JSON - no preamble, no markdown fences, no
commentary. If a figure is genuinely not present in the text, use null
rather than guessing.

JSON shape:
{
  "currency": "USD",
  "unit": "millions",
  "years": {
    "<year>": {
      "revenue": number or null,
      "cost_of_revenue": number or null,
      "operating_income": number or null,
      "net_income": number or null,
      "eps_diluted": number or null,
      "total_current_assets": number or null,
      "total_assets": number or null,
      "total_current_liabilities": number or null,
      "total_liabilities": number or null,
      "total_equity": number or null,
      "cash_and_equivalents": number or null,
      "accounts_receivable": number or null,
      "operating_cash_flow": number or null,
      "capital_expenditures": number or null
    }
  }
}

Rules:
- Numbers must be plain numbers (no $ signs, no commas, no units in the value).
- Negative numbers shown in parentheses in the source, e.g. (32,251), must
  become negative numbers, e.g. -32251.
- Use the exact figures as printed. Do not calculate or estimate anything
  that isn't directly stated in the text.
- "total_equity" is total shareholders'/stockholders' equity (sometimes
  labeled "total stockholders' equity", "total shareholders' equity", or
  "total equity"), NOT total liabilities and equity combined.
"""

# ---------------------------------------------------------------------------
# Locating the real statements inside a 100+ page document
# ---------------------------------------------------------------------------
#
# History of this function, for whoever touches it next:
#
# v1 matched a heading string, then checked whether "years ended" / "as of
# <month>" appeared nearby, on the theory that a table-of-contents mention
# wouldn't have those. That broke on Apple's 10-K: its TOC spells out the
# full date range in the entry itself ("Consolidated Balance Sheets as of
# September 27, 2025 and September 28, 2024 ... 31"), which trips the same
# fingerprint as the real table.
#
# v2 required a units line ("(In millions)", "(in thousands)", etc.)
# immediately after the heading match instead, since a TOC entry never has
# one. That fixed Apple, but broke on Microsoft: a heading string like
# "Consolidated Income Statements" also occurs, verbatim, inside an
# unrelated footnote about an acquisition's pro-forma results - and that
# footnote happens to be followed by its own small "(In millions)" table.
# Because v2 returned on the *first* heading match that had a units line
# nearby, it grabbed that footnote instead of the real income statement.
#
# v3 (this version) stops trusting any single nearby clue and instead
# validates each candidate against what a real statement structurally has
# to contain:
#   1. Anchor terms specific to that statement type (e.g. a real balance
#      sheet must mention both "total assets" and "total liabilities";
#      a real cash flow statement must have an operating section AND an
#      investing-or-financing section - using wording flexible enough to
#      cover "operating activities" as well as Microsoft-style "Operations"
#      / "net cash from operations").
#   2. A minimum density of dollar-formatted numbers in the first ~2000
#      characters. Real statements are dense multi-year tables; footnotes
#      and narrative mentions are not (a genuine income statement has 60+
#      comma-formatted numbers in that space; the Microsoft footnote that
#      fooled v2 had 8).
# Every heading match across every wording variant is checked this way,
# and the candidate with the highest number density wins - so even if a
# decoy passes the anchor check, a real statement elsewhere in the
# document will still outrank it. Only if nothing clears the density bar
# does the code fall back to the weaker anchor-only check, so a genuine
# but sparser statement still gets returned rather than nothing at all.


def _numeric_density(text: str, limit: int = 2000) -> int:
    """Count comma-formatted / parenthesized numbers - a proxy for 'is this
    a real data table' versus narrative prose or a small footnote table."""
    return len(re.findall(r"\d{1,3}(?:,\d{3})+|\(\d{1,3}(?:,\d{3})*\)", text[:limit]))


def _anchors_present(text: str, anchor_groups: list) -> bool:
    """anchor_groups is a list of groups; every group must have at least
    one of its patterns match somewhere in `text` for this to pass."""
    for group in anchor_groups:
        if not any(re.search(pattern, text, re.IGNORECASE) for pattern in group):
            return False
    return True


# Anchor terms are deliberately phrased to cover multiple companies'
# wording, not just one filer's convention.
INCOME_STATEMENT_ANCHORS = [
    [r"net income", r"net loss"],
    [r"total revenue", r"net revenue", r"net sales", r"total net sales", r"\brevenue\b"],
]
BALANCE_SHEET_ANCHORS = [
    [r"total assets"],
    [r"total liabilities"],
]
CASH_FLOW_ANCHORS = [
    [r"cash (?:flows? )?from operating", r"operating activities",
     r"net cash (?:from|used in|provided by) operations?"],
    [r"investing activities", r"financing activities",
     r"net cash (?:from|used in|provided by) (?:investing|financing)"],
]


def find_real_statement(
    full_text: str,
    heading_variations: list,
    anchor_groups: list,
    window: int = 8000,
    min_density: int = 20,
) -> str:
    """
    Different companies title the same statement differently - e.g. Apple
    says "Consolidated Statements of Operations" where Microsoft just says
    "Income Statements." This tries every known wording, evaluates *every*
    match (not just the first), and returns the one that best looks like a
    genuine statement table rather than a table-of-contents entry or an
    incidental footnote mention. See the module-level comment above for why
    this approach replaced two earlier, simpler ones.
    """
    strong_candidates = []
    weak_candidates = []
    for heading in heading_variations:
        pattern = re.compile(re.escape(heading), re.IGNORECASE)
        for match in pattern.finditer(full_text):
            idx = match.start()
            chunk = full_text[idx: idx + window]
            if not _anchors_present(chunk[:3000], anchor_groups):
                continue
            density = _numeric_density(chunk, 2000)
            if density >= min_density:
                strong_candidates.append((density, chunk))
            else:
                weak_candidates.append((density, chunk))

    if strong_candidates:
        strong_candidates.sort(key=lambda c: c[0], reverse=True)
        return strong_candidates[0][1]
    if weak_candidates:
        # Nothing hit the density bar (e.g. a smaller filer with fewer
        # years of data) - still prefer whichever anchor-passing candidate
        # had the most numbers, rather than giving up.
        weak_candidates.sort(key=lambda c: c[0], reverse=True)
        return weak_candidates[0][1]
    return ""  # none of the known variations matched a real table


def build_financial_statements_excerpt(full_text: str) -> tuple:
    """
    Pull just the three core statements out of a much larger document.
    Returns (excerpt_text, missing_statements) so the caller can tell the
    user exactly which statement(s) weren't found, rather than silently
    proceeding with partial data and producing confusing blank fields
    later on.
    """
    income_statement_headings = [
        "Consolidated Statements of Income",
        "Consolidated Statements of Operations",
        "Consolidated Income Statements",
        "Income Statements",
        "Statements of Operations",
        "Income Statement",
    ]
    balance_sheet_headings = [
        "Consolidated Balance Sheets",
        "Consolidated Balance Sheet",
        "Balance Sheets",
        "Balance Sheet",
        "Statements of Financial Position",
        "Statement of Financial Position",
    ]
    cash_flow_headings = [
        "Consolidated Statements of Cash Flows",
        "Consolidated Statement of Cash Flows",
        "Cash Flows Statements",
        "Statements of Cash Flows",
        "Statement of Cash Flows",
        "Cash Flow Statement",
    ]

    statement_types = [
        ("Income Statement", income_statement_headings, INCOME_STATEMENT_ANCHORS),
        ("Balance Sheet", balance_sheet_headings, BALANCE_SHEET_ANCHORS),
        ("Cash Flow Statement", cash_flow_headings, CASH_FLOW_ANCHORS),
    ]

    found_sections = []
    missing_statements = []
    for name, headings, anchors in statement_types:
        section = find_real_statement(full_text, headings, anchors)
        if section:
            found_sections.append(section)
        else:
            missing_statements.append(name)

    excerpt = "\n\n".join(found_sections)
    if not excerpt:
        raise ValueError(
            "Could not locate any of the three core financial statements in this "
            "document using any known heading wording. This company may use "
            "wording we haven't seen yet."
        )
    return excerpt, missing_statements


def extract_financials_from_text(text: str, model: str = "claude-sonnet-4-6") -> dict:
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

    response = client.messages.create(
        model=model,
        max_tokens=2000,
        system=SCHEMA_INSTRUCTIONS,
        messages=[{"role": "user", "content": text}],
    )

    raw_reply = response.content[0].text.strip()

    # Defensive cleanup: sometimes models wrap JSON in ```json fences even when told not to.
    if raw_reply.startswith("```"):
        raw_reply = raw_reply.strip("`")
        raw_reply = raw_reply.replace("json\n", "", 1)

    try:
        return json.loads(raw_reply)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Claude did not return valid JSON. Raw reply was:\n{raw_reply}"
        ) from e


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python extract_financials.py <path_to_clean_text.txt>", file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[1], "r") as f:
        source_text = f.read()

    # Don't feed the AI the entire 100+ page document - that wastes tokens
    # and, as we saw, can miss the actual numbers entirely if they fall
    # outside a blind character slice. Instead, go find the three core
    # financial statements specifically, wherever they are in the document.
    trimmed, missing = build_financial_statements_excerpt(source_text)
    if missing:
        print(f"WARNING: could not find: {', '.join(missing)}", file=sys.stderr)

    result = extract_financials_from_text(trimmed)
    print(json.dumps(result, indent=2))
