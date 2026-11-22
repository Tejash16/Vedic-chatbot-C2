"""C4: rewrite each question into a Devanagari form and an English form with claude-haiku-5-5 (paid).

    python -m src.C.rewrite --estimate     # free token count and cost estimate
    python -m src.C.rewrite                # ../local/work/rewrites.jsonl (+ latency per question)

The rewrites are query text derived from the test questions (no source text). They are kept in
../local/work and summarized in results; the per-question latency feeds the C4 setups.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import time

from src.C.questions import PRICE_IN, PRICE_OUT, Spend, log_spend
from src.common import load as L
from src.common.score import load_questions

MODEL = "claude-haiku-5-5"
OUT = L.WORK / "rewrites.jsonl"
BUDGET_STOP = 0.25
SYSTEM = """You rewrite user questions about the Vedas so a search engine can find the answer in Hindi, Sanskrit \
and English commentaries. The question may be in Hindi (Devanagari or typed in Roman letters), in English, or may \
quote Sanskrit words of a mantra.

Return two forms of the same question:
- devanagari: the question in Hindi, written in Devanagari script. Convert Roman-letter Hindi to Devanagari; \
translate English to Hindi.
- english: the question in English. Write Sanskrit or Hindi terms in Roman letters.
Keep quoted Sanskrit words exactly as quoted, in Devanagari, in both forms. Keep every name and term. Do not answer \
the question and do not add anything."""
SCHEMA = {"type": "object", "properties": {"devanagari": {"type": "string"}, "english": {"type": "string"}},
          "required": ["devanagari", "english"], "additionalProperties": False}


def client():
    import anthropic
    from dotenv import dotenv_values
    return anthropic.Anthropic(api_key=dotenv_values(L.LOCAL / ".env")["VEDIC_ANTHROPIC_API_KEY"], max_retries=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    q = load_questions()
    c = client()
    if a.estimate:
        n = [c.messages.count_tokens(model=MODEL, system=SYSTEM, messages=[{"role": "user", "content": t}]).input_tokens
             for t in q.question.iloc[::30]]
        mean_in = sum(n) / len(n)
        est = len(q) * (mean_in * PRICE_IN + 400 * PRICE_OUT)
        print(f"mean input tokens {mean_in:.0f}; estimate for {len(q)} questions with ~400 output tokens "
              f"(incl. thinking): ${est:.3f} (${est / len(q) * 1000:.2f} per 1,000 questions)")
        return
    spend = Spend()

    def one(row):
        if spend.usd > BUDGET_STOP:
            raise RuntimeError("budget stop")
        t0 = time.perf_counter()
        r = c.messages.create(model=MODEL, max_tokens=1500, system=SYSTEM,
                              messages=[{"role": "user", "content": row.question}],
                              output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}})
        ms = (time.perf_counter() - t0) * 1000
        spend.add(r.usage)
        d = json.loads(next(b.text for b in r.content if b.type == "text"))
        return dict(qid=row.qid, deva=d["devanagari"].strip(), en=d["english"].strip(), ms=ms,
                    in_tokens=r.usage.input_tokens, out_tokens=r.usage.output_tokens)

    t0 = time.time()
    try:
        with cf.ThreadPoolExecutor(a.workers) as ex:
            out = list(ex.map(one, q.itertuples()))
    finally:
        log_spend("C4 rewrites", spend, f"{len(q)} questions, 2 forms each")
    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n", encoding="utf-8")
    print(f"wrote {OUT}: {len(out)} rewrites, ${spend.usd:.4f}, {time.time() - t0:.0f}s, "
          f"${spend.usd / len(out) * 1000:.3f} per 1,000 questions")


def load_rewrites():
    rows = [json.loads(l) for l in OUT.read_text(encoding="utf-8").splitlines() if l.strip()]
    forms = {"deva": {r["qid"]: r["deva"] for r in rows}, "en": {r["qid"]: r["en"] for r in rows}}
    ms = {r["qid"]: r["ms"] for r in rows}
    usd = sum(r["in_tokens"] * PRICE_IN + r["out_tokens"] * PRICE_OUT for r in rows) / len(rows) * 1000
    return forms, ms, usd


if __name__ == "__main__":
    main()
