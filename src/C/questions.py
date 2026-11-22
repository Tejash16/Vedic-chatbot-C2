"""Stage B.3: write the 360 test questions with claude-haiku-5-5 (paid; session C).

    python -m src.C.questions --dry-run     # pick gold chunks, show slots and the cost estimate
    python -m src.C.questions               # generate, validate, write shared/test_questions.jsonl

Gold chunks: corpus chunks with >= 60 tokens, at most one per verse group and 12 per work,
taken in sha256('7|gold|'+chunk_id) order. Every question group gets the same chunk-language
mix (36 Hindi, 32 English, 22 Sanskrit chunks) and the same style mix (direct, reworded, broad).

Checks in code (a failed check retries with feedback, up to 3 tries, then the next chunk):
script of the question, no run of 4+ words copied from the passage, an exact evidence quote
from the passage, and for group sa_quote 2+ consecutive words of the verse.

Evidence quotes are source text: they stay in ../local/work/questions_raw.jsonl, never in Git.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime
import json
import re
import threading
import time
import unicodedata

import pandas as pd

from src.common import load as L
from src.common.chunk import VEDA, h, load_chunks
from src.common.text import words
from src.common.verses import load_verses

MODEL = "claude-haiku-5-5"
PRICE_IN, PRICE_OUT = 0.10 / 1e6, 0.50 / 1e6        # USD per token, Haiku 5.5, prompts < 100k
BUDGET_STOP = 0.45
GROUPS = ["hi", "en", "hi_roman", "sa_quote"]
LANG_QUOTA = {"hi": 36, "en": 32, "sa": 22}
STYLES = ["direct", "reworded", "broad"]
MIN_TOKENS = 60
PER_WORK_CAP = 12
OUT = L.ROOT / "shared" / "test_questions.jsonl"
RAW = L.WORK / "questions_raw.jsonl"
SPEND = L.ROOT / "results" / "C" / "spend.log"
LANG_NAME = {"hi": "Hindi", "en": "English", "sa": "Sanskrit"}

SYSTEM = """You write test questions for a search system over commentaries on Vedic mantras, written in Hindi, \
Sanskrit and English. You get one mantra, its details, and one passage from a commentary on it. Write exactly one \
question that a real user might type to an assistant, and that this passage answers.

Rules:
1. The passage alone must answer the question. Build it on something the passage states clearly.
2. Do not copy the passage's wording. Never reuse a run of three or more consecutive words from the passage; \
single names and Sanskrit terms are fine. Use your own words and your own sentence structure.
3. Do not mention verse numbers, the commentator, the book, or "this passage", "this mantra", "this verse". \
Ask as a user who has not seen the passage.
4. Between 6 and 30 words.
5. Also return `evidence`: the shortest exact quote from the passage (3 to 25 words, copied character for \
character) that answers the question."""

LANG_SPEC = {
    "hi": "Write the question in Hindi, in Devanagari script, in natural modern Hindi. No Roman letters.",
    "en": "Write the question in English. Write Sanskrit or Hindi terms in Roman letters, not Devanagari.",
    "hi_roman": "Write the question in Hindi but typed in Roman (Latin) letters, the way people type on a phone: no "
                "diacritics, casual spelling (kya, hai, ke, mein, kaise, bhagwan, ishwar), mostly lowercase. "
                "No Devanagari at all.",
    "sa_quote_hi": "Write the question in Hindi, in Devanagari. It must quote 2 to 4 consecutive words copied exactly "
                   "from the mantra (not from the passage), the way a user quotes a line they remember.",
    "sa_quote_en": "Write the question in English. It must quote, in Devanagari, 2 to 4 consecutive words copied "
                   "exactly from the mantra (not from the passage), the way a user quotes a line they remember. "
                   "The rest of the question is in English.",
}
STYLE_SPEC = {
    "direct": "Style: direct. Ask about one specific point the passage states.",
    "reworded": "Style: reworded. Ask about one specific point, phrased with different words and a different "
                "structure from the passage, as someone who remembers the idea but not the wording.",
    "broad": "Style: broad. Ask a general question about the theme or teaching that the passage as a whole answers.",
}
SCHEMA = {"type": "object", "properties": {"question": {"type": "string"}, "evidence": {"type": "string"}},
          "required": ["question", "evidence"], "additionalProperties": False}


# ---------------------------------------------------------------- slots and gold chunks
def gold_candidates(ch: pd.DataFrame, verses: pd.DataFrame) -> dict[str, list[str]]:
    vg = dict(zip(verses.verse_id, verses.verse_group))
    el = ch[(ch.n_tokens >= MIN_TOKENS) & ch.lang.isin(LANG_QUOTA)].copy()
    el["vg"] = el.verse_id.map(vg)
    el["key"] = el.chunk_id.map(lambda c: h(f"7|gold|{c}"))
    el = el.sort_values("key")
    used_vg, per_work, out = set(), {}, {k: [] for k in LANG_QUOTA}
    for r in el.itertuples():
        if r.vg in used_vg or per_work.get(r.author_id, 0) >= PER_WORK_CAP:
            continue
        used_vg.add(r.vg)
        per_work[r.author_id] = per_work.get(r.author_id, 0) + 1
        out[r.lang].append(r.chunk_id)
    return out


def make_slots() -> list[dict]:
    slots = []
    """90 slots per group. Chunk languages are interleaved; within each chunk language the styles
    cycle direct/reworded/broad and, for sa_quote, the question language cycles Hindi/English."""
    for g in GROUPS:
        order, counts = [], {k: 0 for k in LANG_QUOTA}
        while len(order) < sum(LANG_QUOTA.values()):
            for k in LANG_QUOTA:
                if counts[k] < LANG_QUOTA[k]:
                    order.append((k, counts[k]))
                    counts[k] += 1
        for i, (lang, k) in enumerate(order):
            q_lang = {"hi": "hi", "en": "en", "hi_roman": "hi-Latn"}.get(g) or ("hi" if k % 2 == 0 else "en")
            slots.append(dict(group=g, slot=i, chunk_lang=lang, style=STYLES[k % 3], q_lang=q_lang))
    return slots


# ---------------------------------------------------------------- checks
def script_share(s: str) -> dict:
    n = {"deva": 0, "latin": 0}
    for ch in s:
        o = ord(ch)
        if 0x0900 <= o <= 0x097F or 0xA8E0 <= o <= 0xA8FF or 0x1CD0 <= o <= 0x1CFF:
            n["deva"] += 1
        elif ch.isalpha() and o < 0x0250:
            n["latin"] += 1
    t = sum(n.values()) or 1
    return {k: v / t for k, v in n.items()}


def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s)).strip()


def ngrams(ws: list[str], n: int) -> set:
    return {tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)}


SOURCE_REFS = re.compile(
    r"\b(this|the above|above|given|that) (mantra|verse|passage|hymn|text|sukta|richa)\b|\bthe (commentary|passage|commentator)\b|"
    r"\baccording to (the|this) (commentary|passage|text|author|commentator)\b|"
    r"\bis (mantra|shlok|richa|sukt)\b|इस\s*(मंत्र|मन्त्र|श्लोक|ऋचा|सूक्त|अंश|पाठ|टीका|भाष्य)|प्रस्तुत|उपर्युक्त|उपरोक्त|टीकाकार|भाष्यकार",
    re.I)


def check(q: str, ev: str, passage: str, verse_text: str, slot: dict) -> str | None:
    """None if the question passes, else the reason."""
    if SOURCE_REFS.search(q):
        return ("do not refer to 'this mantra', 'this verse', 'the passage' or 'the commentary'; ask directly, "
                "as a user who has not seen the text")
    qw = words(q)
    if not 5 <= len(qw) <= 35:
        return f"the question has {len(qw)} words; write 6 to 30"
    sh = script_share(q)
    g = slot["group"]
    if slot["q_lang"] == "hi" and g != "sa_quote" and sh["deva"] < 0.85:
        return "the question must be in Devanagari Hindi only"
    if slot["q_lang"] == "en" and g != "sa_quote" and sh["latin"] < 0.9:
        return "the question must be in English, with Sanskrit terms in Roman letters"
    if g == "hi_roman" and sh["deva"] > 0:
        return "the question must not contain any Devanagari; type Hindi in Roman letters"
    vw = words(verse_text)
    quoted = set()
    if g == "sa_quote":
        hits = ngrams(qw, 2) & ngrams(vw, 2)
        if not hits:
            return "quote 2 to 4 consecutive words copied exactly from the mantra, in Devanagari"
        quoted = {w for t in hits for w in t}
        if slot["q_lang"] == "en" and sh["latin"] < 0.4:
            return "apart from the quoted mantra words, the question must be in English"
    pw = words(passage)
    q_rest = [w for w in qw if w not in quoted]
    if ngrams(q_rest, 4) & ngrams(pw, 4):
        return "the question copies four or more consecutive words from the passage; rephrase in your own words"
    if norm_ws(ev) not in norm_ws(passage) or len(words(ev)) < 2:
        return "the evidence must be an exact quote copied character for character from the passage"
    return None


# ---------------------------------------------------------------- generation
class Spend:
    def __init__(self):
        self.lock = threading.Lock()
        self.inp = self.out = self.calls = 0

    def add(self, u):
        with self.lock:
            self.inp += u.input_tokens
            self.out += u.output_tokens
            self.calls += 1

    @property
    def usd(self):
        return self.inp * PRICE_IN + self.out * PRICE_OUT


def user_prompt(row, verse, slot) -> str:
    lang_key = slot["group"] if slot["group"] != "sa_quote" else f"sa_quote_{slot['q_lang']}"
    det = "; ".join(f"{k}: {verse.get(k)}" for k in ("devata", "rishi", "chhanda") if verse.get(k))
    return (f"<mantra veda=\"{VEDA[verse['collection']]}\">{verse['verse_text']}</mantra>\n"
            f"<mantra_details>{det}</mantra_details>\n"
            f"<passage language=\"{LANG_NAME[row['lang']]}\">{row['text']}</passage>\n\n"
            f"{LANG_SPEC[lang_key]}\n{STYLE_SPEC[slot['style']]}")


def generate_one(client, row, verse, slot, spend: Spend):
    msgs = [{"role": "user", "content": user_prompt(row, verse, slot)}]
    for attempt in range(3):
        if spend.usd > BUDGET_STOP:
            raise RuntimeError("budget stop reached")
        t0 = time.time()
        resp = client.messages.create(model=MODEL, max_tokens=2000, system=SYSTEM, messages=msgs,
                                      output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}})
        spend.add(resp.usage)
        if resp.stop_reason == "refusal":
            return None, "refusal", attempt + 1
        text = next((b.text for b in resp.content if b.type == "text"), "")
        try:
            d = json.loads(text)
        except json.JSONDecodeError:
            msgs += [{"role": "assistant", "content": resp.content},
                     {"role": "user", "content": "Return valid JSON with question and evidence."}]
            continue
        q, ev = d["question"].strip(), d["evidence"].strip()
        why = check(q, ev, row["text"], verse["verse_text"], slot)
        if why is None:
            return dict(question=q, evidence=ev, ms=int((time.time() - t0) * 1000)), None, attempt + 1
        msgs += [{"role": "assistant", "content": resp.content},
                 {"role": "user", "content": f"Not accepted: {why}. Write a new question that follows every rule."}]
    return None, why, 3


def log_spend(step: str, spend: Spend, note: str = ""):
    SPEND.parent.mkdir(parents=True, exist_ok=True)
    line = (f"{datetime.datetime.now().isoformat(timespec='seconds')}\tC\t{step}\t{MODEL}\tcalls={spend.calls}\t"
            f"in={spend.inp}\tout={spend.out}\tusd={spend.usd:.4f}\t{note}\n")
    with open(SPEND, "a", encoding="utf-8") as f:
        f.write(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: only the first N slots of each group")
    a = ap.parse_args()
    global OUT, RAW
    if a.limit:
        OUT, RAW = L.WORK / "test_questions_smoke.jsonl", L.WORK / "questions_raw_smoke.jsonl"
    ch = load_chunks()
    verses = load_verses().set_index("verse_id")
    cands = gold_candidates(ch, verses.reset_index())
    slots = make_slots()
    need = {k: sum(1 for s in slots if s["chunk_lang"] == k) for k in LANG_QUOTA}
    print("candidates per language:", {k: len(v) for k, v in cands.items()}, "needed:", need)
    queues = {k: list(v) for k, v in cands.items()}
    for s in slots:
        s["chunk_id"] = queues[s["chunk_lang"]].pop(0)
    if a.limit:
        slots = [s for s in slots if s["slot"] < a.limit]
    chi = ch.set_index("chunk_id")
    if a.dry_run:
        import anthropic
        from dotenv import dotenv_values
        client = anthropic.Anthropic(api_key=dotenv_values(L.LOCAL / ".env")["VEDIC_ANTHROPIC_API_KEY"])
        sample = [slots[i] for i in range(0, len(slots), 36)]
        toks = []
        for s in sample:
            row = chi.loc[s["chunk_id"]].to_dict()
            row["chunk_id"] = s["chunk_id"]
            n = client.messages.count_tokens(model=MODEL, system=SYSTEM,
                                             messages=[{"role": "user", "content": user_prompt(row, verses.loc[row["verse_id"]].to_dict(), s)}]).input_tokens
            toks.append(n)
        mean_in = sum(toks) / len(toks)
        est = len(slots) * 1.4 * (mean_in * PRICE_IN + 600 * PRICE_OUT)
        print(f"mean input tokens {mean_in:.0f}; estimated cost for {len(slots)} questions with 40% retries "
              f"and ~600 output tokens each: ${est:.2f}")
        print(pd.DataFrame(slots).groupby(["group", "chunk_lang"]).size().unstack().to_string())
        return

    import anthropic
    from dotenv import dotenv_values
    client = anthropic.Anthropic(api_key=dotenv_values(L.LOCAL / ".env")["VEDIC_ANTHROPIC_API_KEY"], max_retries=4)
    spend, lock = Spend(), threading.Lock()
    results = {}

    def work(i):
        s = slots[i]
        tried = []
        cid = s["chunk_id"]
        for _ in range(3):
            row = chi.loc[cid].to_dict()
            row["chunk_id"] = cid
            out, why, n_att = generate_one(client, row, verses.loc[row["verse_id"]].to_dict(), s, spend)
            tried.append((cid, why, n_att))
            if out:
                return i, cid, out, tried
            with lock:
                if not queues[s["chunk_lang"]]:
                    break
                cid = queues[s["chunk_lang"]].pop(0)
        return i, None, None, tried

    t0 = time.time()
    try:
        with cf.ThreadPoolExecutor(a.workers) as ex:
            for i, cid, out, tried in ex.map(work, range(len(slots))):
                results[i] = (cid, out, tried)
                if len(results) % 30 == 0:
                    print(f"{len(results)}/{len(slots)} done, ${spend.usd:.3f}, {time.time() - t0:.0f}s", flush=True)
    finally:
        log_spend("B3 questions", spend, f"{sum(1 for v in results.values() if v[1])} questions accepted")
    rows, raw = [], []
    for i, s in enumerate(slots):
        cid, out, tried = results.get(i, (None, None, []))
        raw.append(dict(slot=s, chunk_id=cid, out=out, tried=tried))
        if not out:
            continue
        c = chi.loc[cid]
        qid = f"q{GROUPS.index(s['group']) + 1}{s['slot']:03d}"
        rows.append(dict(qid=qid, group=s["group"], question=out["question"], gold_chunk_id=cid,
                         record_id=c.record_id, verse_id=c.verse_id, char_start=int(c.char_start),
                         char_end=int(c.char_end), chunk_lang=c.lang, q_lang=s["q_lang"], style=s["style"],
                         verse_group=verses.loc[c.verse_id, "verse_group"], collection=c.collection,
                         author_id=c.author_id, gen_model=MODEL))
    RAW.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in raw) + "\n", encoding="utf-8")
    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    att = [t[2] for v in results.values() for t in v[2]]
    print(f"wrote {OUT}: {len(rows)} questions; calls {spend.calls}; cost ${spend.usd:.4f}; "
          f"first-try accepted {sum(1 for v in results.values() if v[1] and len(v[2]) == 1 and v[2][0][2] == 1)}; "
          f"chunks replaced {sum(1 for v in results.values() if len(v[2]) > 1)}; time {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
