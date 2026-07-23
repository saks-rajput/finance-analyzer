"""
pdf_text_extractor.py

STEP 1 of the pipeline: turn a messy annual-report PDF into clean text.

Why this file exists on its own (and isn't just "read the PDF"):
Real annual report PDFs are inconsistent. Some extract perfectly with a
one-line library call. Others - like the Alphabet 2025 report we tested
this on - use a font encoding where pdfplumber/pdftotext can't map the
character codes back to normal digits, so numbers come out as
"(cid:1727)(cid:1723)" instead of "73". This script detects that problem
and repairs it automatically, so the rest of your pipeline never has to
know or care which kind of PDF it got.

Usage:
    python pdf_text_extractor.py path/to/report.pdf > clean_text.txt
"""

import sys
import re
import pdfplumber

# ---------------------------------------------------------------------------
# The "cid repair" table below was reverse-engineered by cross-checking
# decoded numbers against figures we already knew were correct (e.g. net
# income figures that also appeared, undamaged, elsewhere in the same PDF).
# This exact table is specific to this font subset - if you hit this issue
# on a DIFFERENT company's PDF, you'd need to rebuild the table for that
# PDF's font. That's a real limitation of PDF text extraction worth knowing
# about early: it is not always 100% portable across documents.
# ---------------------------------------------------------------------------
_CID_REPAIR_MAP = {
    1720: '0', 1721: '1', 1722: '2', 1723: '3', 1724: '4',
    1725: '5', 1726: '6', 1727: '7', 1728: '8', 1729: '9',
    1820: ',', 1819: '.', 1921: '$', 1880: ' ', 1821: ':', 1876: '',
}

_CID_PATTERN = re.compile(r'\(cid:(\d+)\)')


def _looks_broken(text: str) -> bool:
    """Heuristic: if we see a bunch of (cid:####) tokens, the font mapping failed."""
    return len(_CID_PATTERN.findall(text)) > 5


def _repair_cid_text(text: str) -> str:
    def repl(match):
        code = int(match.group(1))
        return _CID_REPAIR_MAP.get(code, '')  # unknown codes drop out rather than corrupt the number
    return _CID_PATTERN.sub(repl, text)


def extract_clean_text(pdf_path: str) -> str:
    """Return the full text of the PDF, auto-repairing the cid-encoding quirk if present."""
    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            raw = page.extract_text() or ""
            if _looks_broken(raw):
                raw = _repair_cid_text(raw)
            pages_text.append(raw)
    return "\n".join(pages_text)


def find_section(full_text: str, start_marker: str, end_marker: str) -> str:
    """
    Pull out just the chunk of text between two markers, e.g. between
    'Consolidated Statements of Cash Flows' and 'See accompanying notes'.
    Keeping the AI's input small and focused makes extraction far more
    reliable than dumping the whole 100+ page report into one prompt.
    """
    start = full_text.find(start_marker)
    if start == -1:
        return ""
    end = full_text.find(end_marker, start)
    if end == -1:
        end = start + 4000  # fallback: just grab a reasonable chunk
    return full_text[start:end]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python pdf_text_extractor.py <path_to_pdf>", file=sys.stderr)
        sys.exit(1)

    text = extract_clean_text(sys.argv[1])
    print(text)
