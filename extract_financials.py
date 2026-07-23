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


def find_real_statement(full_text: str, heading: str, window: int = 3500) -> str:
    """
    Annual reports mention statement titles multiple times: once in the
    table of contents, once as the real table, sometimes again in
    narrative text discussing it, and sometimes again in footnotes. A
    loose check like "is there a $ sign nearby" isn't reliable enough -
    ordinary paragraphs can mention dollar figures too, and can end up
    being matched before the real table ever appears.

    A much more reliable fingerprint: every real statement table shows a
    date header ("Year Ended December 31," or "As of December 31,")
    immediately under its title. Plain narrative paragraphs that merely
    reference a statement's name never have this. We check for that
    instead.
    """
    search_from = 0
    while True:
        idx = full_text.find(heading, search_from)
        if idx == -1:
            return ""  # heading not found anywhere in the document
        chunk = full_text[idx: idx + window]
        if "Year Ended" in chunk[:200] or "As of December" in chunk[:200]:
            return chunk
        search_from = idx + len(heading)


def build_financial_statements_excerpt(full_text: str) -> str:
    """Pull just the three core statements out of a much larger document."""
    sections = [
        find_real_statement(full_text, "Consolidated Statements of Income"),
        find_real_statement(full_text, "Consolidated Balance Sheets"),
        find_real_statement(full_text, "Consolidated Statements of Cash Flows"),
    ]
    excerpt = "\n\n".join(s for s in sections if s)
    if not excerpt:
        raise ValueError(
            "Could not locate any of the three core financial statements in this "
            "document. The heading text may be worded differently in this report - "
            "open clean_text.txt and search manually to find the right heading."
        )
    return excerpt


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
