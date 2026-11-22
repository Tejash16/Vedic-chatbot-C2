"""Filter 2: names and Sanskrit terms in a question must appear in the passage or verse details.

Everything is compared through a loose key that works across scripts: Devanagari is
transliterated to plain Latin, aspiration and vowel length are dropped, and a short
inflectional ending is stripped. Then "Agni", "agni", "अग्निम्" and "अग्निः" share one stem.
"""
from __future__ import annotations

import re
import unicodedata

from dutil import norm, tokens

# ------------------------------------------------------------------ transliteration
VOWELS = {
    "अ": "a", "आ": "a", "इ": "i", "ई": "i", "उ": "u", "ऊ": "u", "ऋ": "ri", "ॠ": "ri", "ऌ": "li",
    "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "ऍ": "e", "ऑ": "o",
}
MATRAS = {
    "ा": "a", "ि": "i", "ी": "i", "ु": "u", "ू": "u", "ृ": "ri", "ॄ": "ri", "ॢ": "li",
    "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ॅ": "e", "ॉ": "o",
}
CONS = {
    "क": "k", "ख": "k", "ग": "g", "घ": "g", "ङ": "n", "च": "c", "छ": "c", "ज": "j", "झ": "j", "ञ": "n",
    "ट": "t", "ठ": "t", "ड": "d", "ढ": "d", "ण": "n", "त": "t", "थ": "t", "द": "d", "ध": "d", "न": "n",
    "प": "p", "फ": "p", "ब": "b", "भ": "b", "म": "m", "य": "y", "र": "r", "ल": "l", "ळ": "l", "व": "v",
    "श": "s", "ष": "s", "स": "s", "ह": "h", "ऱ": "r", "ऴ": "l",
}
VIRAMA, NUKTA = "्", "़"
SIGNS = {"ं": "m", "ँ": "n", "ः": "h", "ऽ": ""}


def deva_to_latin(s: str) -> str:
    s = unicodedata.normalize("NFD", s).replace(NUKTA, "")
    s = unicodedata.normalize("NFC", s)
    out, i, n = [], 0, len(s)
    while i < n:
        c = s[i]
        if c in CONS:
            out.append(CONS[c])
            nxt = s[i + 1] if i + 1 < n else ""
            if nxt == VIRAMA:
                i += 2
                continue
            if nxt in MATRAS:
                out.append(MATRAS[nxt])
                i += 2
                continue
            out.append("a")
            i += 1
            continue
        if c in VOWELS:
            out.append(VOWELS[c])
        elif c in MATRAS:
            out.append(MATRAS[c])
        elif c in SIGNS:
            out.append(SIGNS[c])
        elif "ऀ" <= c <= "ॿ":
            pass  # swara marks, digits, other signs
        elif "꣠" <= c <= "ꣿ":
            pass  # Devanagari extended (Vedic accents)
        else:
            out.append(c)
        i += 1
    return "".join(out)


LATIN_RULES = [
    ("ksh", "ks"), ("chh", "c"), ("ch", "c"), ("sh", "s"), ("kh", "k"), ("gh", "g"), ("jh", "j"),
    ("th", "t"), ("dh", "d"), ("ph", "p"), ("bh", "b"), ("w", "v"), ("ee", "i"), ("oo", "u"),
    ("x", "ks"), ("z", "j"), ("q", "k"), ("f", "p"), ("jn", "gy"), ("jny", "gy"),
]


def loose(word: str) -> str:
    """Loose cross-script key of one word."""
    w = norm(word)
    is_deva = any("ऀ" <= c <= "ॿ" for c in w)
    if is_deva:
        w = deva_to_latin(w)
    elif len(w) > 4 and w[-1] in "sS" and w[-2] not in "sS":
        w = w[:-1]                       # English plural: Ashvins, Vasus, Maruts
    w = w.casefold().replace("ṛ", "ri").replace("ṝ", "ri").replace("ḷ", "li")   # IAST vocalic r, l
    w = unicodedata.normalize("NFD", w)
    w = "".join(c for c in w if unicodedata.category(c) != "Mn")
    w = re.sub(r"[^a-z]", "", w)
    for a, b in LATIN_RULES:
        w = w.replace(a, b)
    w = re.sub(r"(.)\1+", r"\1", w)          # aa -> a, tt -> t
    w = re.sub(r"(?<=[^aeiou])h", "", w)    # leftover aspiration
    return w


def stem(key: str) -> str:
    """Strip a case ending: first a final m / h / n after a vowel, then one final vowel; keep at
    least 2 letters. agnim, agnih, agni, agne -> agn; soma, somah -> som; उखा, उखे, Ukha -> uk."""
    if len(key) > 3 and key[-1] in "hmn" and key[-2] in "aeiou":
        key = key[:-1]
    if len(key) > 2 and key[-1] in "aeiou":
        key = key[:-1]
    return key


def skeleton(key: str) -> str:
    """Key without the vowel a, so Hindi schwa deletion does not matter (Satyug ~ सत्ययुग)."""
    return re.sub(r"(.)\1+", r"\1", key.replace("a", ""))


def stem_match(t: str, p: str) -> bool:
    if not t or not p:
        return False
    if len(t) <= 2:
        return p == t
    if p.startswith(t):
        return True
    if t.startswith(p) and len(p) >= max(3, len(t) - 3):
        return True
    ts, ps_ = skeleton(t), skeleton(p)
    return len(ts) >= 4 and ps_.startswith(ts)


# English words of a Latin-script passage and the Devanagari words a Hindi question may use for
# them: names of nature deities that are also plain nouns, and a few common renderings.
EN_TO_DEVA = {
    "fire": "अग्नि", "fires": "अग्नि", "sun": "सूर्य सविता आदित्य", "wind": "वायु", "winds": "वायु मरुत",
    "air": "वायु", "earth": "पृथिवी पृथ्वी भूमि", "moon": "चन्द्रमा चन्द्र", "dawn": "उषा उषस्",
    "dawns": "उषा", "water": "आपः जल", "waters": "आपः जल", "heaven": "द्यौ स्वर्ग", "sky": "आकाश द्यौ",
    "thunderbolt": "वज्र", "speech": "वाक् वाणी", "learned": "विद्वान् विदुषी विद्वानों", "student": "ब्रह्मचारी",
    "celibate": "ब्रह्मचारी", "teacher": "आचार्य गुरु", "preceptor": "आचार्य गुरु", "death": "मृत्यु यम",
    "night": "रात्रि", "rain": "पर्जन्य वर्षा वृष्टि", "cloud": "मेघ पर्जन्य", "clouds": "मेघ पर्जन्य",
    "lightning": "विद्युत्", "ocean": "समुद्र", "sea": "समुद्र", "river": "नदी सिन्धु", "rivers": "नदी सिन्धु",
    "king": "राजा सम्राट्", "guest": "अतिथि", "priest": "होता पुरोहित ऋत्विज्", "sacrifice": "यज्ञ",
    "cow": "गौ गाय", "cows": "गौ गाय", "horse": "अश्व", "horses": "अश्व", "light": "ज्योति", "mind": "मन",
    "breath": "प्राण", "vital": "प्राण", "lord": "प्रभु ईश्वर", "god": "देव परमात्मा परमेश्वर ईश्वर",
    "friend": "मित्र", "plant": "ओषधि वनस्पति", "plants": "ओषधि वनस्पति", "herb": "ओषधि", "herbs": "ओषधि",
    "woman": "स्त्री नारी", "wife": "पत्नी", "husband": "पति", "physician": "वैद्य", "year": "संवत्सर वर्ष",
    "years": "वर्ष", "faith": "श्रद्धा", "strength": "बल", "life": "आयु जीवन", "age": "आयु",
}
EN_TO_DEVA = {k: v.split() for k, v in EN_TO_DEVA.items()}


# ------------------------------------------------------------------ term extraction
HI_STOP = set("""
क्या कौन कौनसा कौनसी कौनसे कैसे कैसा कैसी क्यों किस किसे किसको किसका किसकी किसके किन किनको किनका कहाँ कहां कब कितने कितना कितनी
है हैं था थे थी थीं हो होता होती होते होना होने होकर हुआ हुई हुए गया गई गए गये जाता जाती जाते जाना जाने जा कर करता करती करते करना करने करके किया की गयी
के का की को में से पर और या तथा एवं भी ही तो न नहीं ना इस उस इन उन यह वह ये वे जो जिस जिसे जिन अपने अपना अपनी आप हम मैं तुम तू उसका उसकी उसके इसका इसकी इसके
लिए लिये द्वारा अनुसार बारे संबंध सम्बन्ध विषय बताया बताते बताता बताती कहा कहते कहता कहती कहे गया गयी मन्त्र मंत्र भाष्य भाष्यकार टीका व्याख्या अर्थ वर्णन
प्रकार रूप साथ बिना बाद पहले सब सभी किसी कोई कुछ एक दो तीन क्यूँ वाला वाले वाली रहा रही रहे सकता सकती सकते चाहिए
""".split())

EN_CAP_STOP = set("""
what who whom whose why how when which where whether according in on of the a an is are was were does do did can could should would will
god gods goddess goddesses lord supreme being soul self divine almighty creator truth spirit mantra mantras verse verses hymn hymns veda vedas vedic
commentary commentator commentators author text passage scripture scriptures sanskrit hindi english i o oh my we our he she it they
this that these those there here please tell explain describe give list name say says said by for to from with as and or not
mandala sukta kanda adhyaya chapter book section rigveda samaveda yajurveda atharvaveda
he him his she her hers its they them their thee thou thy thine you your yours one father mother sun moon earth
heaven heavens sky fire water waters wind air dawn night day king queen teacher preceptor master man men woman women
nature lady sir lords omnipresent omniscient almighty
""".split())


# Words of the devata / rishi columns that are generic vocabulary, not names (chosen by hand from
# the name words that are also common in Hindi commentary). Proper deity names stay names.
GENERIC_NAME_WORDS = set("""
वा देवता अथवा आत्मा अध्यात्म देवा देवाः विश्वे विद्या विद्वान् ब्रह्म राजा परमात्मा आप आपः एव पुरुषो मन पवित्र कः नरः प्रजा
सप्त पत्नी अङ्ग मृत्यु अति और विद्युत् नाना ऋषि मनुष्यो कर्म सभा स्वाहा युद्ध होता को पूजा विश्वा सुख आयु प्रकृति माता यज्ञ
मन्त्र भूमि देवः ईश्वरः ईश्वर गृहस्थाः मनुष्याः विद्वांसः वेद वेदः वेदाः
""".split())


class TermChecker:
    """Built once from the corpus; then check(question, grounding_text) per set."""

    def __init__(self, name_words, sanskrit_words, common_hindi_words, hindi_vocab=()):
        # name_words: words of devata/rishi/chhanda names; always checked
        # sanskrit_words: words of verse texts and pada forms; checked unless common in Hindi
        # hindi_vocab: frequent Hindi words; a name stem shared with one of them is ambiguous
        name_words = [w for w in name_words if norm(w).casefold() not in GENERIC_NAME_WORDS]
        self.name_tokens = {norm(w).casefold() for w in name_words}
        generic_stems = {stem(loose(w)) for w in GENERIC_NAME_WORDS}
        clash = {stem(loose(w)) for w in list(hindi_vocab) + list(HI_STOP)
                 if norm(w).casefold() not in self.name_tokens}
        self.name_stems = {stem(loose(w)) for w in name_words if len(stem(loose(w))) >= 3} - generic_stems - clash
        common = {norm(w).casefold() for w in common_hindi_words} | HI_STOP
        self.common = common
        self.sanskrit_tokens = {norm(w).casefold() for w in sanskrit_words} - common
        self.sanskrit_stems = {stem(loose(w)) for w in self.sanskrit_tokens if len(stem(loose(w))) >= 4}

    # -- which words of a question count as a name or a Sanskrit term
    def terms(self, question: str):
        q = norm(question)
        found = []
        # quoted spans are treated as terms (a single quote opens only after a space, so the
        # apostrophe in "woman's" does not); bracketed spans only when they hold Devanagari or
        # IAST, since English brackets usually hold a plain translation such as "(year)"
        quoted = re.findall(r"[\"“]([^\"”]{2,40})[\"”]", q) + \
            re.findall(r"(?:^|(?<=\s))['‘]([^'’]{2,40})['’](?=[\s.,;:?!)]|$)", q)
        for span in quoted:
            found.extend(tokens(span))
        for span in re.findall(r"[(\[]([^)\]]{2,40})[)\]]", q):
            if any("ऀ" <= c <= "ॿ" for c in span) or \
                    any(unicodedata.category(c) == "Mn" for c in unicodedata.normalize("NFD", span)):
                found.extend(tokens(span))
        # words, with hyphenated or slashed compounds split ("Indra-praising" -> Indra, praising);
        # each word carries whether it starts a sentence
        words, sent_start = [], True
        for raw in re.findall(r"\S+", q):
            for pi, part in enumerate(x for x in re.split(r"[-–/]", raw) if x):
                w = re.sub(r"['’]s$", "", part.strip(".,;:!?।()[]\"“”‘")).strip("'’")
                if w:
                    words.append((w, sent_start and pi == 0))
            sent_start = raw[-1] in ".?!:।"
        for w, at_start in words:
            low = w.casefold()
            if any("ऀ" <= c <= "ॿ" for c in w):
                if low in HI_STOP:
                    continue
                if low in self.name_tokens or stem(loose(low)) in self.name_stems:
                    found.append(low)          # names are checked even when common
                elif low in self.common:
                    continue
                elif low.endswith("तः"):
                    continue                   # Hindi adverbs in -तः (अंततः, वस्तुतः, सम्भवतः)
                elif len(low) >= 3 and (low[-1] in "्ः" or "ऽ" in low):
                    # Sanskrit morphology (final virama or visarga, avagraha); plain Hindi
                    # words that merely occur in some verse are not treated as terms
                    found.append(low)
            else:
                # Lowercase English words are not checked: ordinary words collide with name stems
                # (sun, way, pray...). The Hindi twin of the question carries the same terms.
                has_diacritic = any(unicodedata.category(c) == "Mn" for c in unicodedata.normalize("NFD", w))
                if has_diacritic:
                    found.append(low)
                elif w[0].isupper() and not at_start and low not in EN_CAP_STOP and len(stem(loose(w))) >= 3:
                    found.append(low)
        # unique, keep order
        seen, out = set(), []
        for t in found:
            if t not in seen and t not in HI_STOP and t.casefold() not in EN_CAP_STOP:
                seen.add(t)
                out.append(t)
        return out

    @staticmethod
    def grounding_keys(text: str):
        toks = set(tokens(text))
        extra = set()
        for t in toks:                    # Devanagari renderings of English nouns in the passage
            for d in EN_TO_DEVA.get(t, ()):
                extra.add(norm(d).casefold())
        toks |= extra
        return toks, {stem(loose(t)) for t in toks if t}

    def missing(self, question: str, grounding):
        """Terms of the question that appear nowhere in the grounding text."""
        toks, keys = grounding
        miss = []
        for t in self.terms(question):
            if t in toks:
                continue
            k = stem(loose(t))
            if len(k) < 2:
                continue
            if any(stem_match(k, p) for p in keys):
                continue
            miss.append(t)
        return miss
