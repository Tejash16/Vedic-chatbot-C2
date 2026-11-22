"""Loading the Excel library into Parquet tables, with stable IDs.

Every session uses this module, so the same Excel files give the same tables,
record IDs and verse IDs on both laptops.

    python -m src.common.load            # convert ../local/data -> ../local/work/tables

Layout of ../local/data:
    samhitas/<veda>-samhita.xlsx          one row per verse (mantra)
    bhashyas/<vedafolder>/<Work>.xlsx     one row per verse per commentary work

Some anthology works sit, byte-for-byte identical in content, in several Veda
folders. They are loaded once, with collection "multi", and each row's Veda is
inferred from the shape of its reference (see ref_collection).
"""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # the clone
LOCAL = ROOT.parent / "local"
DATA = LOCAL / "data"
WORK = LOCAL / "work"
TABLES = WORK / "tables"

os.environ.setdefault("HF_HOME", str(LOCAL / "hf-cache"))

VEDA_FOLDERS = {"rigved": "rig", "samved": "sama", "yajurved": "yajur", "atharvaved": "atharva"}
SAMHITA_FILES = {"rig": "rigveda-samhita.xlsx", "sama": "samaveda-samhita.xlsx",
                 "yajur": "yajurveda-samhita.xlsx", "atharva": "atharvaveda-samhita.xlsx"}

# Language suffix of the file name -> ISO-like code.
LANGS = [("Hinglish", "hinglish"), ("Sanskrit", "sa"), ("Hindi", "hi"), ("English", "en"),
         ("Bengali", "bn"), ("Gujarati", "gu"), ("Marathi", "mr"), ("Nepali", "ne"),
         ("Tamil", "ta"), ("Urdu", "ur")]

# author_id is the file name as it is (one file = one work); no names are merged.
# author_of() is only a guess at the person, used to list possible matches for the society
# in ../local/final/open-questions.md. It is never used as an ID.
AUTHOR_PATTERNS = [
    ("HarisharanSiddhantalankar", "harisharan_siddhantalankar"),
    ("KshemkaranDasTrivedi", "kshemkarandas_trivedi"),
    ("JaydevSharma", "jaydev_sharma"), ("JoydevSharma", "jaydev_sharma"),
    ("VaidyaDayalmuniArya", "dayalmuni_arya"),
    ("VaidyanathShastri", "vaidyanath_shastri"),
    ("VishwanathVidyalankar", "vishwanath_vidyalankar"),
    ("VishvanathaVidyamartanda", "vishwanath_vidyalankar"),
    ("DeenbandhuVedshashtri", "deenabandhu_vedashastri"),
    ("DeenabandhuVedashastri", "deenabandhu_vedashastri"),
    ("Jambunathan", "mr_jambunathan"),
    ("SatyaPrakashSaraswati", "satyaprakash_saraswati"),
    ("SatyaprakashSaraswati", "satyaprakash_saraswati"),
    ("TulsiramSwami", "tulsiram_swami"),
    ("DrTulsiRam", "tulsi_ram"),
    ("DeviChand", "devi_chand"),
    ("AchyutanandSaraswati", "achyutanand_saraswati"),
    ("AchyutananadSaraswati", "achyutanand_saraswati"),
    ("SudarshandevAcharya", "sudarshandev_acharya"),
    ("DayanandYajurvedabhashyaBhaskar", "sudarshandev_acharya"),
    ("Krishnadutta", "krishnadutt_maharaj"),
    ("HarishchandraVidyalankar", "harishchandra_vidyalankar"),
    ("Aryamuni", "aryamuni"),
    ("ShivShankarSharma", "shivshankar_sharma"),
    ("BrahmamuniParivrajak", "brahmamuni_parivrajak"),
    ("BrahmmuniParivrajak", "brahmamuni_parivrajak"),
    ("KrishnakantVaidikShastri", "krishnakant_vaidik_shastri"),
    ("KrishnakantVedicShastri", "krishnakant_vaidik_shastri"),
    ("RamnathVedalankar", "ramnath_vedalankar"),
    ("VedkumarVedalankar", "vedkumar_vedalankar"),
    ("DharmaDevaVidyaMartanda", "dharmadev_vidyamartand"),
    ("DharmanandSaraswatiVidyaMartand", "dharmadev_vidyamartand"),
    ("DharamadevVidyamartand", "dharmadev_vidyamartand"),
    ("Jagadishwaranand", "jagadishwaranand_saraswati"),
    ("Jagdishwaranand", "jagadishwaranand_saraswati"),
    ("LalitMohanSahni", "lalitmohan_sahni"),
    ("AbhaydevVidyalankar", "abhaydev_vidyalankar"),
    ("SubodhKumar", "subodh_kumar"),
    ("Gurudatt", "gurudatt"),
    ("SatyaBratRajesh", "satyabrat_rajesh"),
    ("AshuRamArya", "ashuram_arya"),
    ("SatishChandraVidyaratna", "satishchandra_vidyaratna"),
    ("MataSavitaJoshi", "savita_joshi"),
    ("Aryaabhivinay", "dayanand_saraswati"),
    ("RigvedadiBhashyaBhumika", "dayanand_saraswati"),
    ("RigvedadibhashyaBhumika", "dayanand_saraswati"),
    ("SwamiDayanand", "dayanand_saraswati"),
    ("RojitChandra", "rojit_chandra"),
]

# Columns that are not commentary text.
META_COLS = {"Mantra Ref ID", "Mandal", "Sukta", "Mantra No", "Kand", "Paryayah", "Adhyay",
             "Mantra Sankhya", "Devata", "Rishi", "Chhanda", "Chhand", "Swara",
             "ऋषि | देवता | छन्द | स्वर", "ऋषि | देवता | छन्द"}
VERSE_TEXT_COLS = {"स्वर रहित मन्त्र", "Mantra with Swara", "मन्त्र (बांग्ला)", "मंत्र", "মন্ত্র"}
PADA_COLS = {"स्वर सहित पद पाठ", "स्वर रहित पद पाठ"}


def column_role(col: str) -> str:
    """'meta', 'verse', 'pada' or 'text' (commentary). Duplicate headers carry a '__n' suffix."""
    base = col.split("__")[0]
    if base in META_COLS:
        return "meta"
    if base in VERSE_TEXT_COLS:
        return "verse"
    if base in PADA_COLS:
        return "pada"
    return "text"


def text_columns(columns) -> list[str]:
    return [c for c in columns if not c.startswith("_") and column_role(c) == "text"]


def lang_of(stem: str) -> str:
    s = re.sub(r"\s*\(\d+\)$", "", stem)
    if s.endswith("fromBengali"):
        return "bn"  # 'Based on ... translation ... from Bengali': the text is Bengali script (audit)
    for suffix, code in LANGS:
        if s.endswith(suffix):
            return code
    return "und"


def author_of(stem: str) -> str:
    for pat, slug in AUTHOR_PATTERNS:
        if pat in stem:
            return slug
    s = re.sub(r"\s*\(\d+\)$", "", stem)
    for suffix, _ in LANGS:
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break
    return re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()


def cell_str(v) -> str:
    """Excel cell -> string. Integral floats lose the '.0'. None -> ''."""
    if v is None:
        return ""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return str(int(v)) if v.is_integer() else repr(v)
    if isinstance(v, int):
        return str(v)
    return str(v)


_REF_CLEAN = re.compile(r"\s+")


def norm_ref(ref: str) -> str:
    """Canonical reference string: digits joined by '/'. '' if it is not a reference."""
    r = _REF_CLEAN.sub("", ref)
    r = re.sub(r"॥\d*॥$", "", r)        # '1/2/3 ॥ 3 ॥' -> '1/2/3'
    if re.fullmatch(r"\d+(/\d+){0,3}", r):
        return "/".join(str(int(x)) for x in r.split("/"))
    return ""


def ref_collection(nref: str) -> list[str]:
    """Possible Vedas for a reference shape. 1 part: Sama; 2: Yajur; 3: Rig or Atharva; 4: Atharva."""
    n = nref.count("/") + 1 if nref else 0
    return {1: ["sama"], 2: ["yajur"], 3: ["rig", "atharva"], 4: ["atharva"]}.get(n, [])


def list_files() -> list[dict]:
    """Every Excel file, with its table id. Duplicated anthologies are listed once."""
    out: list[dict] = []
    for coll, fname in SAMHITA_FILES.items():
        p = DATA / "samhitas" / fname
        out.append(dict(path=p, kind="samhita", collection=coll, stem=p.stem, lang="sa",
                        author_id="samhita", table_id=f"samhita.{coll}", copies=[str(p.relative_to(DATA))]))
    by_name: dict[str, list[Path]] = {}
    for folder in sorted(VEDA_FOLDERS):
        for p in sorted((DATA / "bhashyas" / folder).glob("*.xlsx")):
            by_name.setdefault(p.name, []).append(p)
    for name in sorted(by_name):
        paths = by_name[name]
        p = paths[0]
        multi = len(paths) > 1
        coll = "multi" if multi else VEDA_FOLDERS[p.parent.name]
        stem = p.stem                      # the file name as it is: one file = one work
        lang = lang_of(stem)
        out.append(dict(path=p, kind="bhashya", collection=coll, stem=stem, lang=lang,
                        author_id=stem, table_id=f"{coll}.{stem}",
                        copies=[str(x.relative_to(DATA)) for x in paths]))
    return out


def read_rows(path: Path) -> list[list]:
    from python_calamine import CalamineWorkbook
    wb = CalamineWorkbook.from_path(str(path))
    if len(wb.sheet_names) != 1:
        raise ValueError(f"{path}: expected one sheet, found {wb.sheet_names}")
    return wb.get_sheet_by_name(wb.sheet_names[0]).to_python()


def dedupe_headers(hdr: list) -> list[str]:
    out, seen = [], {}
    for h in hdr:
        h = cell_str(h).strip() or "unnamed"
        if h in seen:
            seen[h] += 1
            h = f"{h}__{seen[h]}"
        else:
            seen[h] = 1
        out.append(h)
    return out


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def convert_all(verbose=True):
    """Write one Parquet per table plus tables/_index.parquet. All cells are strings, unmodified
    except that integral floats lose '.0'. Column _row is the Excel row number (header = row 1).
    Column _raw_type_ref keeps the raw type of the reference cell."""
    import pandas as pd
    TABLES.mkdir(parents=True, exist_ok=True)
    index = []
    for f in list_files():
        rows = read_rows(f["path"])
        hdr = dedupe_headers(rows[0])
        width = len(hdr)
        recs, n_empty = [], 0
        for i, r in enumerate(rows[1:], start=2):
            r = list(r) + [None] * (width - len(r))
            if all(c is None or (isinstance(c, str) and c.strip() == "") for c in r):
                n_empty += 1
                continue
            d = {h: cell_str(c) for h, c in zip(hdr, r[:width])}
            d["_row"] = i
            d["_raw_type_ref"] = type(r[0]).__name__
            recs.append(d)
        df = pd.DataFrame(recs, columns=hdr + ["_row", "_raw_type_ref"])
        df.to_parquet(TABLES / f"{f['table_id']}.parquet", index=False)
        index.append(dict(table_id=f["table_id"], kind=f["kind"], collection=f["collection"],
                          stem=f["stem"], lang=f["lang"], author_id=f["author_id"],
                          path=str(f["path"].relative_to(DATA)), copies=";".join(f["copies"]),
                          n_rows=len(df), n_empty_rows=n_empty, n_cols=len(hdr),
                          sha256=file_sha256(f["path"])))
        if verbose:
            print(f"{f['table_id']}: {len(df)} rows")
    pd.DataFrame(index).to_parquet(TABLES / "_index.parquet", index=False)
    return index


def load_index():
    import pandas as pd
    return pd.read_parquet(TABLES / "_index.parquet")


def load_table(table_id: str):
    import pandas as pd
    return pd.read_parquet(TABLES / f"{table_id}.parquet")


if __name__ == "__main__":
    convert_all()
