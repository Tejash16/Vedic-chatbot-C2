"""Session D helpers: paths, text normalization, tokenizing, sentence split, spend log.

No source text is ever written under the repo. Everything derived from the corpus goes to
../local/work/D (work files) or ../local/final (reports for humans).
"""
from __future__ import annotations

import datetime
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

SEED = 7
ROOT = Path(__file__).resolve().parents[2]          # the clone root (session-D)
LOCAL = ROOT.parent / "local"
WORK = LOCAL / "work"
WORK_D = WORK / "D"
FINAL = LOCAL / "final"
RESULTS_D = ROOT / "results" / "D"
SPEND_LOG = RESULTS_D / "spend.log"

HAIKU = "claude-haiku-5-5"
SONNET = "claude-sonnet-5-5"
# US dollars per million tokens, standard rate (prompts up to 100K tokens). Batch API = 50 %.
PRICES = {HAIKU: (0.10, 0.50), SONNET: (2.00, 10.00)}
BATCH_DISCOUNT = 0.5

os.environ.setdefault("HF_HOME", str(LOCAL / "hf-cache"))


def utf8_stdout():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def api_key() -> str:
    """Read VEDIC_ANTHROPIC_API_KEY from ../local/.env without exporting anything."""
    from dotenv import dotenv_values
    vals = dotenv_values(LOCAL / ".env")
    key = (vals.get("VEDIC_ANTHROPIC_API_KEY") or "").strip()
    if not key:
        sys.exit("VEDIC_ANTHROPIC_API_KEY is missing from ../local/.env")
    return key


def usd(model: str, in_tok: int, out_tok: int, batch: bool) -> float:
    pin, pout = PRICES[model]
    f = BATCH_DISCOUNT if batch else 1.0
    return f * (in_tok * pin + out_tok * pout) / 1e6


def log_spend(step: str, model: str, n_requests: int, in_tok: int, out_tok: int, batch: bool, note: str = ""):
    RESULTS_D.mkdir(parents=True, exist_ok=True)
    cost = usd(model, in_tok, out_tok, batch)
    line = "%s | %s | %s | requests=%d | in_tokens=%d | out_tokens=%d | batch=%s | usd=%.4f | %s\n" % (
        datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        step, model, n_requests, in_tok, out_tok, "yes" if batch else "no", cost, note)
    with open(SPEND_LOG, "a", encoding="utf-8") as f:
        f.write(line)
    return cost


def total_spend() -> float:
    if not SPEND_LOG.exists():
        return 0.0
    tot = 0.0
    for line in SPEND_LOG.read_text(encoding="utf-8").splitlines():
        m = re.search(r"usd=([0-9.]+)", line)
        if m:
            tot += float(m.group(1))
    return tot


# ---------------------------------------------------------------- normalization

ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)
PUNCT_MAP = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-",
    " ": " ", "|": "।", "॥": "।।",
})


def norm(s: str) -> str:
    """NFC, no zero-width characters, unified quotes/dashes/dandas, single spaces."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFC", str(s)).translate(ZERO_WIDTH).translate(PUNCT_MAP)
    return re.sub(r"\s+", " ", s).strip()


def _is_sep(c: str) -> bool:
    if c.isspace():
        return True
    cat = unicodedata.category(c)
    return cat[0] in "PSZ" or c in "।॥"


def tokens_with_spans(s: str):
    """Split on whitespace, punctuation and symbols (never inside a word: Devanagari vowel
    signs and viramas stay attached). Returns [(token_casefolded, start, end)] over s."""
    out, start = [], None
    for i, c in enumerate(s):
        if _is_sep(c):
            if start is not None:
                out.append((s[start:i].casefold(), start, i))
                start = None
        elif start is None:
            start = i
    if start is not None:
        out.append((s[start:].casefold(), start, len(s)))
    return out


def tokens(s: str):
    return [t for t, _, _ in tokens_with_spans(norm(s))]


def find_quote(passage: str, quote: str):
    """Word-for-word match after normalization: the quote's token sequence must appear as a
    contiguous run of the passage's tokens. Returns (char_start, char_end) in norm(passage)
    or None."""
    p = norm(passage)
    pt = tokens_with_spans(p)
    qt = tokens(quote)
    if not qt or len(qt) > len(pt):
        return None
    ptoks = [t for t, _, _ in pt]
    n = len(qt)
    first = qt[0]
    for i in range(len(ptoks) - n + 1):
        if ptoks[i] == first and ptoks[i:i + n] == qt:
            return pt[i][1], pt[i + n - 1][2]
    return None


SENT_END = re.compile(r"(?<=[.!?।])\s+|(?<=।)(?=\S)")


def sentence_spans(text: str):
    """Sentence spans over norm(text). Ends: . ! ? and the danda (। and ॥ after norm)."""
    p = norm(text)
    spans, start = [], 0
    i, n = 0, len(p)
    while i < n:
        c = p[i]
        if c in ".!?।":
            j = i + 1
            while j < n and p[j] in ".!?।\"')]":
                j += 1
            # skip decimals / references like 1.2.3
            if c == "." and j < n and p[j].isdigit():
                i = j
                continue
            if p[start:j].strip():
                spans.append((start, j))
            while j < n and p[j].isspace():
                j += 1
            start = i = j
            continue
        i += 1
    if p[start:].strip():
        spans.append((start, n))
    # drop spans that hold no word characters (stray dandas, numbers only)
    return [(a, b) for a, b in spans if any(not _is_sep(ch) and not ch.isdigit() for ch in p[a:b])]


def n_words(text: str) -> int:
    return len(norm(text).split())


def script_of(s: str) -> str:
    deva = sum(1 for c in s if "ऀ" <= c <= "ॿ")
    latin = sum(1 for c in s if c.isascii() and c.isalpha())
    if deva == 0 and latin == 0:
        return "other"
    return "deva" if deva >= latin else "latin"


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def machine_string() -> str:
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU"
    except Exception:
        gpu = "unknown GPU"
    try:
        import psutil
        ram = "%.0fGB RAM" % (psutil.virtual_memory().total / 1e9)
    except Exception:
        ram = ""
    return "Tejash laptop: %s, %d CPU threads, %s, Windows" % (gpu, os.cpu_count() or 0, ram)
