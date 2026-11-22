"""Write ../local/final/data-issues.csv and ../local/final/open-questions.md (session C, local only).

    python -m src.C.data_issues

data-issues.csv holds work names, Excel rows and references only, never source text.
"""
from __future__ import annotations

import collections
import csv
import re

from src.common import load as L
from src.common.config import config

FINAL = L.LOCAL / "final"
DEVANAGARI_OR_BENGALI = re.compile(r"[ऀ-৿]")


def ref_shown(raw: str) -> str:
    raw = raw.strip()
    if DEVANAGARI_OR_BENGALI.search(raw) and not re.fullmatch(r"[\d/\s॥०-९]+", raw):
        return "(cell holds text, not a reference)"
    return raw[:40]


def issues() -> list[dict]:
    tables = config()["tables"]
    samhita = {c: set(x for x in L.load_table(f"samhita.{c}")["Mantra Ref ID"].map(L.norm_ref) if x)
               for c in L.SAMHITA_FILES}
    rows = []

    def add(issue, tid, excel_row, ref, column="", detail=""):
        spec = tables[tid]
        rows.append(dict(issue=issue, work=spec["author_id"] or tid, table_id=tid, file=";".join(spec["copies"]),
                         excel_row=excel_row, reference=ref, column=column, detail=detail))

    for tid, spec in tables.items():
        if spec["kind"] != "bhashya":
            continue
        df = L.load_table(tid)
        seen = collections.defaultdict(list)
        for r in df.to_dict("records"):
            raw, row = r["Mantra Ref ID"], int(r["_row"])
            ref = L.norm_ref(raw)
            if not ref:
                add("unreadable_reference", tid, row, ref_shown(raw), "Mantra Ref ID",
                    "reference cell is not of the form n/n/n")
                continue
            seen[ref].append(row)
            cands = L.ref_collection(ref) if spec["collection"] == "multi" else [spec["collection"]]
            hits = [c for c in cands if ref in samhita[c]]
            if len(hits) > 1:
                add("unclear_veda", tid, row, ref, "Mantra Ref ID",
                    "anthology row; reference exists in both " + " and ".join(hits))
            elif not hits:
                add("verse_missing", tid, row, ref, "Mantra Ref ID",
                    "no verse with this reference in the " + "/".join(cands) + " samhita table")
        for ref, rws in seen.items():
            if len(rws) > 1:
                for rw in rws:
                    add("duplicate_reference_in_work", tid, rw, ref, "Mantra Ref ID",
                        f"same reference on Excel rows {', '.join(map(str, rws))}")
        for c in [c for c in df.columns if not c.startswith("_")]:
            for r in df[df[c].str.len() >= 32767].to_dict("records"):
                add("cell_cut_at_32767", tid, int(r["_row"]), L.norm_ref(r["Mantra Ref ID"]), c,
                    "cell length is exactly 32,767 characters, Excel's limit: text is probably cut")
    y = L.load_table("samhita.yajur")
    y["n"] = y["Mantra Ref ID"].map(L.norm_ref)
    for ref in ("25/47", "25/48"):
        r = y[y.n == ref].iloc[0]
        add("yajurveda_extra_row", "samhita.yajur", int(r["_row"]), ref, "स्वर रहित मन्त्र",
            "1,976 rows against the usual 1,975: the verse text of 25/47 already contains the first words of 25/48; "
            "the pada text numbers them 47 and 48; Satyaprakash's English has no 25/48. One mantra (25.47 in "
            "editions with 1,975) is probably split into two rows. To confirm with the society.")
    return rows


OPEN_QUESTIONS = """\
# Open questions for the society

Written by session C during the C2 benchmark. Nothing here has been merged or changed in the data.
Every file is treated as its own work, and `author_id` is the file name as it is.

## 1. Two pairs of names that may be the same person

| File names | Why they may be one person |
|---|---|
| `SamVedaAdhyatmikBhashyaPdtVishvanathaVidyamartandaHindi` (and its Bengali translation) and `AtharvavedaBhashyaVishwanathVidyalankarHindi` | Pt. Vishwanath Vidyalankar is also known as Vidyamartanda; both are spiritual (adhyatmik) commentaries in the same style. |
| `SamvedaAcharyaDharmaDevaVidyaMartandaEnglish`, `TheRigvedaBasedontranslationofSwamiDayanandSwamiDharmanandSaraswatiVidyaMartandEnglish` and `RigvedBhashyaShubhankarMandal...SwamiDharamadevVidyamartandfromBengali` | Acharya Dharmadeva Vidya Martanda is believed to have taken sannyasa as Swami Dharmananda Saraswati; the title "Vidya Martand" appears in all three. |

Please confirm or correct each pair.

## 2. Spelling variants of the same name across file names (also not merged)

{variants}

## 3. Yajurveda verse table: 1,976 rows

The usual count is 1,975. Mantras 25/47 and 25/48 look like one mantra split into two rows: the verse text of
25/47 already holds the first words of 25/48, while the pada text numbers them 47 and 48. Some commentaries
(Dayanand, Tulsi Ram) have both numbers; Swami Satyaprakash's English has no 25/48. Which numbering should we use?

## 4. Same mantra in several Vedas

For the benchmark, verses are grouped when their unaccented text is identical after removing accents, spaces and
punctuation. By that rule 406 Samaveda verses match a Rigveda verse. If the Samaveda sign ꣳ is read as anusvara (ं)
and the avagraha (ऽ) is ignored, 629 match. Should those spellings count as the same text?

## 5. Placeholder text

Cells with "missing" (16,231), "N/A" (5,799), "\\"Missing\\"" (587), "NA" and "x" are treated as empty. One cell in
`VedSandeshHindi` holds test text ("adfasdf"); that file is left out.
"""


def variants_table() -> str:
    tables = config()["tables"]
    by = collections.defaultdict(list)
    for tid, spec in tables.items():
        if spec["kind"] == "bhashya":
            by[(L.author_of(spec["author_id"]), spec["lang"])].append(spec["author_id"])
    lines = ["| Probable person (guess) | Language | File names |", "|---|---|---|"]
    for (who, lang), stems in sorted(by.items()):
        stems = sorted(set(stems))
        if len(stems) > 1:
            lines.append(f"| {who} | {lang} | " + "<br>".join(f"`{s}`" for s in stems) + " |")
    return ("These files look like the same commentator writing on different Vedas, sometimes with the name "
            "spelled differently (e.g. Jaydev/Joydev, Brahmamuni/Brahmmuni, Achyutanand/Achyutananad, "
            "Deenbandhu/Deenabandhu, Krishnakant Vaidik/Vedic). They are kept as separate works.\n\n" + "\n".join(lines))


def main():
    FINAL.mkdir(parents=True, exist_ok=True)
    rows = issues()
    with open(FINAL / "data-issues.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote data-issues.csv:", dict(collections.Counter(r["issue"] for r in rows)))
    (FINAL / "open-questions.md").write_text(OPEN_QUESTIONS.replace("{variants}", variants_table()), encoding="utf-8")
    print("wrote open-questions.md")


if __name__ == "__main__":
    main()
