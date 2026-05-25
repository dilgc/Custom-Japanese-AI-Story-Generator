from fugashi import Tagger
from config import CFORM_EXPLANATIONS, CTYPE_EXPLANATIONS

_tagger = None


def _get_tagger() -> Tagger:
    global _tagger
    if _tagger is None:
        _tagger = Tagger()
    return _tagger


def _safe_feature(word, attr: str) -> str:
    try:
        val = getattr(word.feature, attr)
        return str(val) if val and str(val) != "*" else ""
    except Exception:
        return ""


def tokenize_story(text: str) -> list[dict]:
    tagger = _get_tagger()
    tokens = []
    for word in tagger(text):
        surface = str(word)
        lemma = _safe_feature(word, "lemma") or surface
        pos1 = _safe_feature(word, "pos1")
        pos2 = _safe_feature(word, "pos2")
        pos = f"{pos1}-{pos2}" if pos2 else pos1

        c_type = _safe_feature(word, "cType")
        c_form = _safe_feature(word, "cForm")
        reading = _safe_feature(word, "kana")

        token = {
            "surface": surface,
            "lemma": lemma,
            "pos": pos1,
            "pos_full": pos,
            "cType": c_type,
            "cType_en": CTYPE_EXPLANATIONS.get(c_type, c_type),
            "cForm": c_form,
            "cForm_en": CFORM_EXPLANATIONS.get(c_form, c_form),
            "reading": reading,
        }
        tokens.append(token)
    return tokens
