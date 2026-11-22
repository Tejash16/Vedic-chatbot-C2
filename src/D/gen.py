"""Stage D step 1: generate question-and-answer sets with Claude.

One request per passage. The model first lists every distinct point of the passage as an
exact quote, then writes one set per point (Hindi question, English question). The answer of
a set is the point's quote, filled in by code.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor

from dutil import HAIKU, SONNET, api_key, log_spend, usd

SYSTEM = """You prepare question-and-answer sets for a search index over commentaries on Vedic mantras. Users will ask questions in Hindi or English; each set you write lets the search find the passage that answers it.

You receive one PASSAGE (part of one author's commentary on one mantra) and CONTEXT about the mantra. Work in two steps.

Step 1 - points. List every distinct point the PASSAGE makes, in passage order. A point is one statement of meaning, teaching, instruction or fact. Give each point as an exact quote from the PASSAGE:
- copy a contiguous span character for character, in the passage's own script and spelling;
- no paraphrase, no added or removed words, no ellipsis, no joining of separate spans;
- usually one clause or one sentence; a run of word-by-word glosses that together state one idea is one point;
- cover the whole passage, and do not list the same point twice.
There is no fixed number of points: a short passage may have one or two, a dense one many.

Step 2 - sets. Write exactly one set for each point:
- question_hi: a natural question in Hindi (Devanagari script) that the quote answers;
- question_en: the same question in English.
Rules for the questions:
- ask the way a reader would, in your own words; do not copy the quote's wording;
- make the question specific to this passage: name its subject, so that it would not fit any other passage equally well;
- use only names and Sanskrit terms that appear in the PASSAGE or the CONTEXT; never introduce a deity, sage, place or term that is not there;
- do not mention "the passage", "the author" or "the commentary"; a question may mention the mantra.

Return JSON only, matching the schema. "sets" has exactly as many items as "points", in the same order; "point" is the 1-based number of the point the set answers."""

SCHEMA = {
    "type": "object",
    "properties": {
        "points": {"type": "array", "items": {"type": "string"}},
        "sets": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "point": {"type": "integer"},
                    "question_hi": {"type": "string"},
                    "question_en": {"type": "string"},
                },
                "required": ["point", "question_hi", "question_en"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["points", "sets"],
    "additionalProperties": False,
}

EFFORT = {HAIKU: "low", SONNET: "low"}
MAX_TOKENS = 8000


def user_message(p: dict) -> str:
    ctx = [
        "Verse reference: %s" % (p.get("verse_ref") or ""),
        "Collection: %s" % (p.get("collection") or ""),
        "Author: %s" % (p.get("author") or ""),
    ]
    for label, key in (("Mantra", "mantra"), ("Pada form", "pada"), ("Devata", "devata"),
                       ("Rishi", "rishi"), ("Chhanda", "chhanda")):
        if p.get(key):
            ctx.append("%s: %s" % (label, p[key]))
    return "CONTEXT\n%s\n\nPASSAGE\n%s" % ("\n".join(ctx), p["text"])


def params(model: str, p: dict) -> dict:
    return dict(
        model=model,
        max_tokens=MAX_TOKENS,
        system=SYSTEM,
        output_config={"effort": EFFORT[model], "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": user_message(p)}],
    )


def parse_message(msg) -> dict:
    """Return {'points': [...], 'sets': [...], 'in_tokens', 'out_tokens', 'stop', 'error'}."""
    out = {"points": [], "sets": [], "in_tokens": 0, "out_tokens": 0, "stop": None, "error": None}
    u = getattr(msg, "usage", None)
    if u is not None:
        out["in_tokens"] = (u.input_tokens or 0) + (getattr(u, "cache_read_input_tokens", 0) or 0) \
            + (getattr(u, "cache_creation_input_tokens", 0) or 0)
        out["out_tokens"] = u.output_tokens or 0
    out["stop"] = msg.stop_reason
    text = next((b.text for b in msg.content if b.type == "text"), "")
    if msg.stop_reason not in ("end_turn", "stop_sequence"):
        out["error"] = "stop_reason=%s" % msg.stop_reason
    try:
        data = json.loads(text)
        out["points"] = [str(x) for x in data.get("points", [])]
        out["sets"] = data.get("sets", [])
    except Exception as e:  # truncated or empty
        out["error"] = (out["error"] or "") + " json:%s" % type(e).__name__
    return out


def client():
    import anthropic
    return anthropic.Anthropic(api_key=api_key(), max_retries=4)


def run_sync(model: str, passages: list, step: str, workers: int = 4) -> dict:
    """Plain Messages API, a few threads. Used for pilots. Returns {chunk_id: parsed}."""
    cl = client()

    def one(p):
        for attempt in range(3):
            try:
                return p["chunk_id"], parse_message(cl.messages.create(**params(model, p)))
            except Exception as e:
                err = "%s: %s" % (type(e).__name__, str(e)[:200])
                time.sleep(2 + 3 * attempt)
        return p["chunk_id"], {"points": [], "sets": [], "in_tokens": 0, "out_tokens": 0,
                               "stop": None, "error": err}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        res = dict(ex.map(one, passages))
    tin = sum(r["in_tokens"] for r in res.values())
    tout = sum(r["out_tokens"] for r in res.values())
    cost = log_spend(step, model, len(passages), tin, tout, batch=False)
    print("%s: %d requests, %d in / %d out tokens, $%.4f" % (step, len(passages), tin, tout, cost))
    return res


def submit_batch(model: str, passages: list):
    """Returns (batch_id, id_map custom_id -> chunk_id)."""
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    cl = client()
    id_map = {"p%05d" % i: p["chunk_id"] for i, p in enumerate(passages)}
    reqs = [Request(custom_id="p%05d" % i, params=MessageCreateParamsNonStreaming(**params(model, p)))
            for i, p in enumerate(passages)]
    b = cl.messages.batches.create(requests=reqs)
    return b.id, id_map


def wait_batch(batch_id: str, poll: int = 30, log=print):
    cl = client()
    while True:
        b = cl.messages.batches.retrieve(batch_id)
        c = b.request_counts
        log("batch %s: %s | processing=%d succeeded=%d errored=%d" % (
            batch_id, b.processing_status, c.processing, c.succeeded, c.errored))
        if b.processing_status == "ended":
            return b
        time.sleep(poll)


def collect_batch(batch_id: str, model: str, id_map: dict, step: str) -> dict:
    """id_map: custom_id -> chunk_id. Returns {chunk_id: parsed}."""
    cl = client()
    res = {}
    for r in cl.messages.batches.results(batch_id):
        cid = id_map.get(r.custom_id, r.custom_id)
        if r.result.type == "succeeded":
            res[cid] = parse_message(r.result.message)
        else:
            res[cid] = {"points": [], "sets": [], "in_tokens": 0, "out_tokens": 0, "stop": None,
                        "error": "batch:%s" % r.result.type}
    tin = sum(r["in_tokens"] for r in res.values())
    tout = sum(r["out_tokens"] for r in res.values())
    cost = log_spend(step, model, len(res), tin, tout, batch=True, note="batch %s" % batch_id)
    print("%s: %d results, %d in / %d out tokens, $%.4f" % (step, len(res), tin, tout, cost))
    return res


def estimate_cost(model: str, passages: list, out_per_passage: int, batch: bool) -> dict:
    """Pre-run estimate. Input tokens from count_tokens on a sample, output from a guess."""
    cl = client()
    sample = passages[:: max(1, len(passages) // 20)][:20]
    counts = []
    for p in sample:
        pr = params(model, p)
        r = cl.messages.count_tokens(model=model, system=pr["system"], messages=pr["messages"],
                                     output_config=pr["output_config"])
        counts.append(r.input_tokens)
    avg_in = sum(counts) / len(counts)
    n = len(passages)
    return {"n": n, "avg_in": avg_in, "out_per_passage": out_per_passage,
            "usd": usd(model, int(avg_in * n), int(out_per_passage * n), batch)}


def fake_generate(passages: list) -> dict:
    """Dry run without paid calls: deterministic made-up sets, including deliberate failures."""
    import random
    from dutil import sentence_spans, norm
    rnd = random.Random(7)
    res = {}
    for p in passages:
        text = norm(p["text"])
        sents = [text[a:b] for a, b in sentence_spans(text)]
        points, sets = [], []
        for i, s in enumerate(sents):
            words = s.split()
            r = rnd.random()
            quote = s
            if r < 0.10 and len(words) > 4:      # paraphrased quote -> filter 1
                quote = " ".join(words[:2] + ["वास्तव", "में"] + words[2:])
            points.append(quote)
            subj = " ".join(words[: min(5, len(words))])
            qhi = "%s के विषय में क्या कहा गया है?" % subj
            qen = "What is said about %s?" % subj
            if 0.10 <= r < 0.18:                  # hallucinated name -> filter 2
                qen = "What does Vishvamitra say about %s?" % subj
                qhi = "विश्वामित्र %s के विषय में क्या कहते हैं?" % subj
            if 0.18 <= r < 0.25:                  # generic question -> likely filter 3
                qen, qhi = "What is the meaning?", "इसका अर्थ क्या है?"
            sets.append({"point": i + 1, "question_hi": qhi, "question_en": qen})
            if r > 0.92:                          # a repeat -> merge
                sets.append({"point": i + 1, "question_hi": qhi + " ", "question_en": qen})
        res[p["chunk_id"]] = {"points": points, "sets": sets, "in_tokens": 900, "out_tokens": 120 * len(sets),
                              "stop": "end_turn", "error": None}
    return res
