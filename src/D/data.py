"""Inputs for Stage D: passages with context, test questions, verse details, scoring.

Real data comes from session C (../local/work/chunks.parquet, shared/test_questions.jsonl,
src/common). The dry-run data below is made up for testing the code and holds no corpus text.
"""
from __future__ import annotations

import sys

from dutil import ROOT

sys.path.insert(0, str(ROOT))          # session C's code imports as src.common

# ------------------------------------------------------------------ dry-run data (made up)
_FAKE = [
    ("hi", "प्रभु सबका रक्षक है। जो मनुष्य सत्य का पालन करता है वह कभी भयभीत नहीं होता। यज्ञ से वातावरण शुद्ध होता है और मन में शान्ति आती है।"),
    ("hi", "विद्वान् लोग प्रातःकाल उठकर ध्यान करते हैं। ध्यान से बुद्धि तीव्र होती है। आलस्य मनुष्य का सबसे बड़ा शत्रु है।"),
    ("hi", "माता-पिता की सेवा सबसे बड़ा धर्म है। जो सन्तान सेवा करती है उसे सुख मिलता है।"),
    ("hi", "अग्नि प्रकाश का स्रोत है। अग्नि के बिना भोजन नहीं पकता। इसलिए अग्नि को देवताओं का मुख कहा गया है। वैद्य अग्नि-तत्त्व को पाचन से जोड़ते हैं।"),
    ("en", "The seeker who speaks the truth gains the trust of all. Truthful speech is described as the foundation of a peaceful society. Lies destroy friendship."),
    ("en", "Water is called the giver of life. Rivers carry nourishment to the fields. One should never pollute flowing water."),
    ("en", "A king must protect the weak. He should collect taxes fairly, like the sun drawing water. A just ruler is loved by the people."),
    ("en", "Knowledge removes darkness as the lamp removes night. The teacher lights the lamp of wisdom in the pupil."),
    ("hi", "दान देने से धन घटता नहीं बढ़ता है। सुपात्र को दिया गया दान श्रेष्ठ है। दान में अहंकार नहीं होना चाहिए।"),
    ("hi", "सूर्य जगत् की आत्मा है। सूर्य के प्रकाश से रोग दूर होते हैं।"),
    ("en", "Indra is praised as the destroyer of obstacles. The worshipper asks for strength to overcome inner enemies such as anger and greed."),
    ("hi", "इन्द्र बल का प्रतीक है। उपासक इन्द्र से प्रार्थना करता है कि वह काम और क्रोध को जीत सके।"),
]


def fake_passages():
    out = []
    for i, (lang, text) in enumerate(_FAKE):
        vid = "V%02d" % (i // 2)
        out.append({
            "chunk_id": "fake%03d" % i, "record_id": "R%03d" % i, "verse_id": vid,
            "author_id": "A%d" % (i % 3), "author": "Author %d" % (i % 3), "collection": "TestVeda",
            "lang": lang, "header": "TestVeda %s | Author %d" % (vid, i % 3), "verse_ref": vid, "text": text,
            "mantra": "अग्निम् ईळे पुरोहितम्" if i < 4 else "इन्द्रम् वर्धन्तः",
            "pada": "अग्निम् । ईळे । पुरोहितम्" if i < 4 else "इन्द्रम् । वर्धन्तः",
            "devata": "अग्निः" if i < 4 else "इन्द्रः", "rishi": "मधुच्छन्दाः", "chhanda": "गायत्री",
        })
    return out


def fake_questions():
    qs = [
        ("q1", "hindi", "सत्य का पालन करने वाला किससे नहीं डरता?", "fake000"),
        ("q2", "english", "Why is truthful speech important for society?", "fake004"),
        ("q3", "roman", "dhyan karne se buddhi par kya asar hota hai", "fake001"),
        ("q4", "english", "Who removes darkness like a lamp?", "fake007"),
        ("q5", "hindi", "दान के विषय में क्या कहा गया है?", "fake008"),
        ("q6", "sanskrit", "इन्द्रम् वर्धन्तः का भाव क्या है?", "fake011"),
    ]
    ps = {p["chunk_id"]: p for p in fake_passages()}
    return [{"qid": a, "group": g, "question": q, "gold_chunk_id": c, "verse_id": ps[c]["verse_id"]}
            for a, g, q, c in qs]


def fake_authors():
    return [
        {"author_id": "A0", "author": "Author 0", "lang": "hi", "script": "deva", "words": 1_200_000, "records": 9000,
         "chars": 6_000_000, "chunks": 10_000},
        {"author_id": "A1", "author": "Author 1", "lang": "hi", "script": "deva", "words": 400_000, "records": 3000,
         "chars": 2_000_000, "chunks": 3500},
        {"author_id": "A2", "author": "Author 2", "lang": "en", "script": "latin", "words": 250_000, "records": 2500,
         "chars": 1_400_000, "chunks": 2600},
    ]


def fake_lexicon():
    names = ["अग्निः", "इन्द्रः", "मधुच्छन्दाः", "गायत्री", "विश्वामित्रः", "वसिष्ठः"]
    sanskrit = ["अग्निम्", "ईळे", "पुरोहितम्", "इन्द्रम्", "वर्धन्तः", "सोमः", "वरुणः"]
    common = []
    return names, sanskrit, common, []


# ------------------------------------------------------------------ real data (session C)
TARGET_CHUNKS = 1500
PER_VERSE_CAP = 10      # whole verses average 11.5 chunks; a cap of 10 fits 165 questions in 1,500 chunks


def real_available() -> bool:
    return (ROOT / "src" / "common").exists() and (ROOT / "shared" / "test_questions.jsonl").exists()


def chunks_df():
    from src.common.chunk import load_chunks
    return load_chunks()


def load_questions():
    from src.common.score import load_questions as lq
    q = lq()
    return q.to_dict("records")


def questions_df():
    from src.common.score import load_questions as lq
    return lq()


def verses_df():
    from src.common.verses import load_verses
    return load_verses()


def index_text(p):
    from src.common.chunk import embed_text
    return embed_text(p, header=True)


def verse_ref_of(header: str) -> str:
    return (header or "").split(" | ")[0]


def work_of(header: str) -> str:
    parts = (header or "").split(" | ", 1)
    return parts[1] if len(parts) > 1 else ""


def select_subcorpus(work):
    """~1,500 chunks made of verses, as in the 10k corpus: the commentary chunks of each chosen
    verse compete with each other. Only verses holding a test question are eligible; they are
    added in a seed-7 random order while the total stays within 1,500. A verse with more than
    PER_VERSE_CAP chunks keeps its gold chunk and a seeded sample of the others."""
    import json
    import numpy as np
    from dutil import RESULTS_D, SEED, write_jsonl

    ch = chunks_df().sort_values("chunk_id").reset_index(drop=True)
    q = questions_df()
    v = verses_df().set_index("verse_id")
    gold = set(q.gold_chunk_id)
    gold_verses = sorted(set(ch.loc[ch.chunk_id.isin(gold), "verse_id"]))
    by_verse = {vid: sorted(g.chunk_id) for vid, g in ch.groupby("verse_id")}
    rng = np.random.RandomState(SEED)
    order = rng.permutation(gold_verses)
    picked, keep, total = [], [], 0
    for vid in order:
        ids = by_verse[vid]
        if len(ids) > PER_VERSE_CAP:     # the gold chunk(s) plus a seeded sample of the others
            g = [c for c in ids if c in gold]
            others = [c for c in ids if c not in gold]
            ids = sorted(g + list(rng.choice(others, PER_VERSE_CAP - len(g), replace=False)))
        if total + len(ids) <= TARGET_CHUNKS:
            picked.append(vid)
            keep.extend(ids)
            total += len(ids)
        if total >= TARGET_CHUNKS - 5:
            break
    sub = ch[ch.chunk_id.isin(set(keep))].copy()
    qs = q[q.gold_chunk_id.isin(sub.chunk_id)]
    passages = []
    for r in sub.to_dict("records"):
        vd = v.loc[r["verse_id"]] if r["verse_id"] in v.index else None
        passages.append({
            "chunk_id": r["chunk_id"], "record_id": r["record_id"], "verse_id": r["verse_id"],
            "author_id": r["author_id"], "author": work_of(r["header"]), "collection": r["collection"],
            "lang": r["lang"], "header": r["header"], "verse_ref": verse_ref_of(r["header"]),
            "text": r["text"], "n_tokens": int(r["n_tokens"]),
            "mantra": "" if vd is None else vd["verse_text"],
            "pada": "" if vd is None else vd["pada"],
            "devata": "" if vd is None else vd.get("devata", ""),
            "rishi": "" if vd is None else vd.get("rishi", ""),
            "chhanda": "" if vd is None else vd.get("chhanda", ""),
            "verse_group": "" if vd is None else vd["verse_group"],
        })
    write_jsonl(work / "subcorpus.jsonl", passages)
    RESULTS_D.mkdir(parents=True, exist_ok=True)
    (RESULTS_D / "subcorpus_chunk_ids.txt").write_text("\n".join(sorted(sub.chunk_id)) + "\n", encoding="utf-8")
    info = {
        "chunks": len(sub), "verses": len(picked), "questions": len(qs),
        "questions_by_group": qs.group.value_counts().to_dict(),
        "questions_by_chunk_lang": qs.chunk_lang.value_counts().to_dict(),
        "chunks_by_lang": sub.lang.value_counts().to_dict(),
        "chunks_by_collection": sub.collection.value_counts().to_dict(),
        "works": int(sub.author_id.nunique()),
        "verses_with_questions_total": len(gold_verses),
    }
    (work / "subcorpus_info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(json.dumps(info, indent=1))
    return passages


def load_lexicon(work, df_threshold=0.01, vocab_threshold=0.001):
    """Names (devata, rishi, chhanda words), Sanskrit words (verse text and pada words) from all
    verses, common Hindi words (in >= 1 % of the Hindi chunks of the 10k corpus) and the wider
    Hindi vocabulary (>= 0.1 %), used to drop name stems that clash with ordinary words."""
    import json
    from collections import Counter
    from dutil import tokens
    path = work / "lexicon.json"
    if path.exists():
        d = json.loads(path.read_text(encoding="utf-8"))
        if "vocab" in d:
            return d["names"], d["sanskrit"], d["common"], d["vocab"]
    v = verses_df()
    names, sanskrit = set(), set()
    for col in ("devata", "rishi", "chhanda"):
        for s in v[col].dropna():
            names.update(t for t in tokens(s) if len(t) > 1)
    for col in ("verse_text", "pada"):
        for s in v[col].dropna():
            sanskrit.update(t for t in tokens(s) if len(t) > 1)
    ch = chunks_df()
    hi = ch[ch.lang == "hi"].text
    df = Counter()
    for t in hi:
        df.update(set(tokens(t)))
    common = sorted(w for w, c in df.items() if c >= df_threshold * len(hi))
    vocab = sorted(w for w, c in df.items() if c >= vocab_threshold * len(hi))
    d = {"names": sorted(names), "sanskrit": sorted(sanskrit), "common": common, "vocab": vocab}
    path.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    print("lexicon: %d name words, %d Sanskrit words, %d common Hindi words, %d vocabulary" % (
        len(names), len(sanskrit), len(common), len(vocab)))
    return d["names"], d["sanskrit"], d["common"], d["vocab"]


SCRIPT_OF_LANG = {"hi": "deva", "sa": "deva", "mr": "deva", "ne": "deva", "en": "latin", "hinglish": "latin"}


def load_author_totals(work):
    """Whole dataset (every language): words, records and estimated requests per work.
    Requests = chunks of ~300 bge-m3 tokens, estimated per record as ceil(tokens / 300) with
    tokens from the audit's bge-m3 tokens-per-character ratio for the record's language group."""
    import json
    import math
    path = work / "author_totals.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(ROOT.parent / "local" / "work" / "records.parquet")
    bge_per_char = {"en": 0.279, "hi": 0.345, "sa": 0.414}   # audit section 7; 'other' = 0.359
    rows = {}

    def records():                        # streamed in batches to keep memory low
        for b in pf.iter_batches(batch_size=5000, columns=["author_id", "lang", "text"]):
            d = b.to_pydict()
            yield from zip(d["author_id"], d["lang"], d["text"])

    for aid, lang, text in records():
        r = rows.setdefault(aid, {"author_id": aid, "author": aid, "lang": lang,
                                  "script": SCRIPT_OF_LANG.get(lang, "other"), "words": 0, "records": 0,
                                  "chunks": 0, "chars": 0})
        r["words"] += len(text.split())
        r["records"] += 1
        r["chars"] += len(text)
        r["chunks"] += max(1, math.ceil(len(text) * bge_per_char.get(lang, 0.359) / 300))
    out = sorted(rows.values(), key=lambda r: -r["words"])
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out
