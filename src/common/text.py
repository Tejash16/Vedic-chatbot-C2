"""Text cleaning, record text and sentence boundaries. Shared by every session.

The cleaning fixes what the Stage A audit found: placeholder cells ('N/A', 'NULL', ...),
'<br>' tags, a stray control character, NBSP, invisible characters and mixed Unicode forms.
"""
from __future__ import annotations

import re
import unicodedata

# Placeholder cells found by the Stage A audit ('missing' 16k, 'N/A' 5.8k, '"Missing"' 587, 'NA', 'x').
PLACEHOLDERS = {"n/a", "na", "null", "none", "undefined", "nan", "-", "—", "missing", '"missing"', "x"}
_BR = re.compile(r"<\s*br\s*/?\s*>", re.I)
_TAG = re.compile(r"</?(?:p|div|span|b|i|u|strong|em|font|sup|sub|a|li|ul|ol|table|tr|td|th|h\d|img)\b[^>]*>", re.I)
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_INVISIBLE = dict.fromkeys(map(ord, "​﻿⁠­"), None)   # ZWSP, BOM, WJ, soft hyphen; ZWJ/ZWNJ are kept
_SPACES = re.compile(r"[ 	  -  　]+")
_MANY_NL = re.compile(r"\n{3,}")


def clean(s: str) -> str:
    """One cell -> clean NFC text. Placeholder cells become ''."""
    if not s:
        return ""
    if s.strip().lower() in PLACEHOLDERS:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _BR.sub("\n", s)
    s = _TAG.sub("", s)
    s = _CTRL.sub("", s)
    s = s.translate(_INVISIBLE)
    s = unicodedata.normalize("NFC", s)
    s = "\n".join(_SPACES.sub(" ", line).strip() for line in s.split("\n"))
    s = _MANY_NL.sub("\n\n", s)
    s = s.strip()
    return "" if s.lower() in PLACEHOLDERS else s


FIELD_SEP = "\n\n"


def record_text(cells: list[str]) -> str:
    """Commentary cells of one record, in sheet order, cleaned and joined by a blank line."""
    return FIELD_SEP.join(c for c in (clean(x) for x in cells) if c)


# A sentence ends after a danda or double danda (with an optional verse number such as
# '॥१॥' or '। 3 ।'), after ? or !, after a full stop followed by space (not a list number
# such as '७.' or '3.'), or at a line break.
_SENT_END = re.compile(
    r"(?:[।॥]+(?:\s*[0-9०-९]+\s*[।॥]+)?|[?!]+|(?<![0-9०-९])\.(?=\s)|\n+)"
)


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) of each sentence, without surrounding whitespace. Covers all non-space text."""
    spans, start = [], 0
    for m in _SENT_END.finditer(text):
        end = m.end()
        a, b = start, end
        while a < b and text[a].isspace():
            a += 1
        while b > a and text[b - 1].isspace():
            b -= 1
        if b > a:
            spans.append((a, b))
        start = end
    a, b = start, len(text)
    while a < b and text[a].isspace():
        a += 1
    while b > a and text[b - 1].isspace():
        b -= 1
    if b > a:
        spans.append((a, b))
    return spans


# Word characters: \w plus combining marks of the Indic blocks and Vedic extensions.
# The danda (U+0964), double danda (U+0965) and abbreviation sign (U+0970) are separators.
_WORD = re.compile(r"[\ẁ-ͯऀ-ॣ०-९ॱ-ॿঀ-৿"
                   r"઀-૿᳐-᳿꣠-ꣿ]+")


def words(text: str) -> list[str]:
    """Split on spaces, punctuation and danda after NFC, never inside a word (the C1a tokenizer).
    Combining marks (matras, virama, nukta, Vedic accents) stay inside the word."""
    return [w.lower() for w in _WORD.findall(unicodedata.normalize("NFC", text))]
