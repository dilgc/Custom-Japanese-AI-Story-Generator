import threading
import unicodedata

import streamlit as st
from jamdict import Jamdict

_local = threading.local()


def _get_jam() -> Jamdict:
    if not hasattr(_local, "jam"):
        _local.jam = Jamdict()
    return _local.jam


def _kata_to_hira(text: str) -> str:
    """Convert katakana to hiragana for alternate lookup attempts."""
    return "".join(
        chr(ord(c) - 0x60) if "ァ" <= c <= "ン" else c
        for c in text
    )


def _candidates(lemma: str) -> list[str]:
    """Return lookup candidates to try in order."""
    cands = [lemma]
    # する-verb: also try the noun stem (e.g. 両断する → 両断)
    if lemma.endswith("する"):
        cands.append(lemma[:-2])
    # ずる-verb: try noun stem (e.g. 生ずる → 生)
    if lemma.endswith("ずる"):
        cands.append(lemma[:-2])
    # い-adjective inflected: try base
    if lemma.endswith("い") and len(lemma) > 1:
        cands.append(lemma)
    return cands


def _extract(result, lemma: str) -> dict | None:
    if not result.entries:
        return None
    entry = result.entries[0]
    definitions, pos_tags = [], []
    for sense in entry.senses:
        glosses = [str(g) for g in sense.gloss
                   if not hasattr(g, "lang") or g.lang in (None, "eng")]
        if glosses:
            definitions.append("; ".join(glosses))
        pos_tags.extend([str(p) for p in sense.pos])
    if not definitions:
        return None
    return {
        "word": lemma,
        "kanji_forms": [str(k) for k in entry.kanji_forms],
        "readings": [str(k) for k in entry.kana_forms],
        "definitions": definitions,
        "pos": list(dict.fromkeys(pos_tags)),  # deduplicate, preserve order
    }


def lookup_word(lemma: str) -> dict:
    jam = _get_jam()

    for candidate in _candidates(lemma):
        result = jam.lookup(candidate)
        hit = _extract(result, lemma)
        if hit:
            return hit

    # Last resort: wildcard on original lemma
    result = jam.lookup(f"%{lemma}%")
    hit = _extract(result, lemma)
    if hit:
        return hit

    return {"word": lemma, "definitions": [], "pos": [], "readings": [], "kanji_forms": []}


@st.cache_data(show_spinner=False)
def cached_lookup(lemma: str) -> dict:
    return lookup_word(lemma)
