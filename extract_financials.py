"""
extract_financials.py

STEP 2 of the pipeline: turn clean text into structured numbers.

This is the one piece that actually uses AI. Everything before it
(pdf_text_extractor.py) and everything after it (a future ratio_engine.py)
is plain, deterministic code - no AI, fully traceable. This is deliberate:
we only want AI doing the part it's genuinely good at (reading messy
prose/tables and mapping them to a schema), never the arithmetic.

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
"""


def find_real_statement(full_text: str, heading_variations: list, window: int = 3500) -> str:
    """
    Different companies title the same statement differently - e.g. Apple
    says "Consolidated Statements of Operations" where Alphabet says
    "Consolidated Statements of Income." This function tries a whole list
    of known variations, case-insensitively, and returns the first real
    match it finds (using the same "does it have a date header nearby"
    fingerprint as before to skip table-of-contents mentions).
    """
    for heading in heading_variations:
        pattern = re.compile(re.escape(heading), re.IGNORECASE)
        search_from = 0
        while True:
            match = pattern.search(full_text, search_from)
            if not match:
                break  # this wording not found, try the next variation
            idx = match.start()
            chunk = full_text[idx: idx + window]
            fingerprint = chunk[:250].lower()
            if any(marker in fingerprint for marker in
                   ["year ended", "years ended", "as of december", "as of june",
                    "as of september", "fiscal year ended", "months ended"]):
                return chunk
            search_from = idx + len(heading)
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
    ]
    balance_sheet_headings = [
        "Consolidated Balance Sheets",
        "Consolidated Balance Sheet",
        "Balance Sheets",
    ]
    cash_flow_headings = [
        "Consolidated Statements of Cash Flows",
        "Consolidated Statement of Cash Flows",
        "Cash Flows Statements",
        "Statements of Cash Flows",
    ]

    statement_types = [
        ("Income Statement", income_statement_headings),
        ("Balance Sheet", balance_sheet_headings),
        ("Cash Flow Statement", cash_flow_headings),
    ]

    found_sections = []
    missing_statements = []
    for name, headings in statement_types:
        section = find_real_statement(full_text, headings)
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
    trimmed = build_financial_statements_excerpt(source_text)

    result = extract_financials_from_text(trimmed)
    print(json.dumps(result, indent=2))
