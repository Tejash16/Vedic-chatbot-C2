"""Write shared/config.yaml (session C only).

    python -m src.C.make_config

Every table gets an explicit list of commentary columns (names only), the verse-detail
columns, and the excluded columns with the reason. The defaults come from the column
headers; the overrides below come from reading the columns during Stage A/B.
"""
from __future__ import annotations

import collections

import yaml

from src.common import load as L
from src.common.text import clean

CONFIG = L.ROOT / "shared" / "config.yaml"

DEITY_SEER_METRE = {"Devata", "Rishi", "Chhanda", "Chhand", "Swara",
                    "ऋषि | देवता | छन्द | स्वर", "ऋषि | देवता | छन्द"}
VERSE_NUMBER = {"Mantra Ref ID", "Mandal", "Sukta", "Mantra No", "Kand", "Paryayah", "Adhyay", "Mantra Sankhya"}

KK = "rig.RigvedabhashyamSubodhBhashyaDrKrishnakantVaidikShastriHindi"
# (table_id, column) -> ("commentary" | "verse_details" | "excluded", reason or None, column language or None)
OVERRIDES = {
    (KK, "संस्कृत भाग"): ("excluded", "verse text, pada text and a copy of Dayanand's Sanskrit commentary in one cell", None),
    (KK, "सन्धिविच्छेदसहितोऽन्वयः"): ("commentary", None, "sa"),
    (KK, "विशेष"): ("commentary", "about 2,000 cells hold only seer/deity/metre lines; those cells are dropped by cell_rules", None),
    ("sama.SamVedaSwamiSatyaprakashSaraswatiEnglish", "Notes"): ("excluded", "cross-references to other verses only ('Cf. Rv ...')", None),
    ("atharva.AtharvaVedaSwamiSatyaPrakashSaraswatiEnglish", "Comments / Notes"): ("excluded", "editorial notes on numbering ('MANTRA NO ... AS PER THE BOOK')", None),
    ("atharva.AtharvavedaBhashyamPanditKshemkaranDasTrivediSanskrit", "सूचना"): ("excluded", "editorial pointer: 'see the Hindi bhashya'", None),
    ("sama.SamvedaAcharyaDharmaDevaVidyaMartandaEnglish", "Comments"): ("commentary", None, "sa"),
    ("sama.SamvedBhashyaSwamiBrahmamuniParivrajakHindi", "विशेष"): ("verse_details", "seer, deity and metre lines", None),
    ("yajur.DayanandYajurvedaBhashyaBhaskarPtSudarshandevAcharyaMAHindi", "विशेष"): ("verse_details", "seer, deity and metre lines", None),
    ("yajur.DayanandYajurvedaBhashyaBhaskarPtSudarshandevAcharyaMASanskrit", "विशेषः"): ("verse_details", "seer, deity and metre lines", None),
    ("multi.VedaadhyayanPraveshikaSwamiBrahmmuniParivrajakHindi", "विशेष"): ("verse_details", "seer, deity and metre lines", None),
    ("multi.vedonkemotiHindi", "सूक्त"): ("excluded", "a quoted phrase of the verse (verse text)", None),
    ("multi.vedonkemotiHindi", "Purport"): ("commentary", None, "en"),
    ("multi.vedonkemotiHindi", "Sukta Meaning"): ("commentary", None, "en"),
    ("rig.VedasmadeEasyEnglish", "Literal meaning"): ("commentary", None, "sa"),
    ("rig.AnthologyofVedicHymnsEnglish", "Anvaya"): ("commentary", None, "sa"),
}
EXCLUDED_TABLES = {"rig.VedSandeshHindi": "test data only ('adfasdf')"}

# Cells in a commentary column that are verse details, not commentary.
CELL_RULES = {
    KK: {"विशेष": {"drop_if_matches": r"(?s)^(?=.{0,60}ऋषि)(?=.{0,150}देवता)|^\s*शाक्तः\s+देवता|^\s*ऋषि[ः:]?\s*[-–—]|^\s*छन्दः\s*[-–—]",
                   "reason": "seer/deity/metre lines"}},
}


def classify(tid: str, col: str, table_lang: str):
    if tid in EXCLUDED_TABLES:
        return "excluded", EXCLUDED_TABLES[tid], None
    if (tid, col) in OVERRIDES:
        return OVERRIDES[(tid, col)]
    base = col.split("__")[0]
    if base in VERSE_NUMBER:
        return "excluded", "verse reference or number", None
    if base in DEITY_SEER_METRE:
        return "verse_details", "seer, deity, metre or note", None
    if base in L.VERSE_TEXT_COLS:
        return "excluded", "verse text", None
    if base in L.PADA_COLS:
        return "excluded", "pada text", None
    return "commentary", None, None


def build_tables() -> dict:
    idx = L.load_index()
    out = {}
    for t in idx.itertuples():
        df = L.load_table(t.table_id)
        cols = [c for c in df.columns if not c.startswith("_")]
        entry = dict(file=t.path.replace("\\", "/"), copies=t.copies.replace("\\", "/").split(";"),
                     kind=t.kind, collection=t.collection, lang=t.lang,
                     author_id=t.author_id if t.kind == "bhashya" else None,
                     rows=int(t.n_rows))
        if t.kind == "samhita":
            entry.update(verse_text="स्वर रहित मन्त्र", verse_text_accented="Mantra with Swara",
                         pada="स्वर रहित पद पाठ", pada_accented="स्वर सहित पद पाठ",
                         verse_details=[c for c in cols if c in DEITY_SEER_METRE],
                         number_columns=[c for c in cols if c in VERSE_NUMBER])
            out[t.table_id] = entry
            continue
        commentary, col_lang, details, excluded = [], {}, [], {}
        for c in cols:
            kind, reason, lang = classify(t.table_id, c, t.lang)
            if kind == "commentary":
                filled = sum(1 for v in df[c] if clean(v))
                if filled == 0:
                    excluded[c] = "empty after cleaning"
                    continue
                commentary.append(c)
                if lang and lang != t.lang:
                    col_lang[c] = lang
            elif kind == "verse_details":
                details.append(c)
            else:
                excluded[c] = reason
        entry.update(commentary=commentary, verse_details=details, excluded=excluded)
        if col_lang:
            entry["column_lang"] = col_lang
        if t.table_id in CELL_RULES:
            entry["cell_rules"] = CELL_RULES[t.table_id]
        out[t.table_id] = entry
    return out


SETTINGS = {
    "seed": 7,
    "paths": {"data": "../local/data", "work": "../local/work", "final": "../local/final",
              "hf_home": "../local/hf-cache", "env": "../local/.env (VEDIC_ANTHROPIC_API_KEY)"},
    "text": {"normalization": "NFC", "placeholders": ["N/A", "NA", "NULL", "None", "undefined", "nan", "-", "missing", '"Missing"', "x"],
             "applied_to": "commentary, verse text and pada text",
             "record_text": "commentary columns in sheet order, cleaned, joined by a blank line; no column names"},
    "chunking": {"tokenizer": "BAAI/bge-m3", "target_tokens": 300, "variants_C6": [150, 600],
                 "rule": "break at a column boundary first, then at sentence ends (danda, double danda, ? ! and full stop + space, line break); "
                         "small whole columns of the same language share a chunk; a chunk smaller than 0.3 x target takes the next column's first sentences; "
                         "a last chunk smaller than 0.3 x target joins the one before if both fit in 1.3 x target; a sentence longer than target is cut at a space",
                 "header": "'<Veda> <ref> | <work name from the file name>' (no column names); embed header + newline + text unless the setup removes the header",
                 "max_length_passages": 512, "max_length_600_variant": 1024},
    "corpus": {"languages": ["hi", "sa", "en"], "target_chunks": 10000,
               "selection": "verses ordered by (rank within their Veda / verses in that Veda), rank = sha256('7|'+verse_id); "
                            "take every hi/sa/en record of each verse until 10,000 chunks; data-only and deterministic",
               "full": "every hi/sa/en chunk (session C runs it once as corpus_size 'full')"},
    "verse_group": "NFC unaccented verse text with accent marks, spaces, punctuation and digits removed; verses with the same key form a group; "
                   "verse-level scoring counts any verse of the gold verse's group",
    "questions": {"model": "claude-haiku-5-5",
                  "groups": {"hi": "Hindi in Devanagari", "en": "English", "hi_roman": "Hindi typed in Roman letters, phone style",
                             "sa_quote": "Hindi or English question quoting a few Sanskrit words of the verse"},
                  "per_group": 90, "gold_chunk_langs_per_group": {"hi": 36, "en": 32, "sa": 22},
                  "styles": ["direct", "reworded", "broad"], "real": "group 'real' may be added later; always reported separately"},
    "candidates": {
        "keyword": {"C1a": "bm25s, own tokenizer (src.common.text.words)", "C1b": "OpenSearch BM25, Hindi analyzer for hi/sa, English analyzer for en, plus unstemmed field"},
        "embedding": {"C2a": "BAAI/bge-m3 dense, sparse, dense+sparse", "C2b": "Qwen/Qwen3-Embedding-0.6B dense, alone and fused with C1a"},
        "fusion": {"method": "reciprocal rank fusion", "k": 60},
        "question_forms": {"C4": "as typed; claude-haiku-5-5 rewrite to Devanagari and to English; each fused with the original"},
        "rerankers": {"C5": ["BAAI/bge-reranker-v2-m3", "Qwen/Qwen3-Reranker-0.6B", "claude-haiku-5-5 picks best 8"], "keep": 8,
                      "input": "shared/candidates_top50.jsonl"},
        "chunk_variants": {"C6": ["150 tokens", "600 tokens", "300 tokens without header"], "embedding": "bge-m3 dense+sparse",
                           "gold": "any chunk of the gold record overlapping the gold character span"},
    },
    "gpu": {"precision": "fp16", "lock": "python bus/bus.py gpu acquire|release"},
    "metrics": {"k": [5, 10, 20], "mrr_at": 10, "ndcg_at": 10, "levels": ["chunk", "verse"],
                "verse_level": "a hit is any chunk whose verse is in the gold verse's group",
                "chunk_level": "a hit is the gold chunk only (C6: overlap rule above)",
                "latency": "p50 and p95 per question for the stage, on that machine, in ms",
                "cost": "paid US dollars per 1,000 questions"},
    "results_schema": ["session", "stage", "run_id", "setup", "corpus_size", "question_group", "level", "recall_5", "recall_10",
                       "recall_20", "mrr_10", "ndcg_10", "p50_ms", "p95_ms", "paid_usd_per_1000", "machine", "notes"],
    "results_optional": ["recall_8", "mrr_8", "ndcg_8"],
}


def main():
    tables = build_tables()
    cfg = dict(SETTINGS)
    cfg["tables"] = tables
    CONFIG.parent.mkdir(exist_ok=True)
    header = ("# C2 benchmark configuration. Written by session C (python -m src.C.make_config). Do not edit by hand.\n"
              "# tables.<id>.commentary is the exact list of commentary columns, in sheet order.\n")
    CONFIG.write_text(header + yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False, width=200), encoding="utf-8")
    n = collections.Counter()
    for e in tables.values():
        n["commentary"] += len(e.get("commentary", []))
        n["verse_details"] += len(e.get("verse_details", []))
        n["excluded"] += len(e.get("excluded", {}))
    print("wrote", CONFIG, dict(n))


if __name__ == "__main__":
    main()
