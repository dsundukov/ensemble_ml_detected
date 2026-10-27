import re
import unicodedata

import pandas as pd

_INVISIBLE_RE = re.compile("[\u00ad\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff]")
_WORD_RE = re.compile(r"\w+")

_LAT = "aceopxyACEHKMOPTXB"
_CYR = "асеорхуАСЕНКМОРТХВ"
_LAT2CYR = str.maketrans(_LAT, _CYR)
_CYR2LAT = str.maketrans(_CYR, _LAT)


def _is_cyr(ch: str) -> bool:
    return "\u0400" <= ch <= "\u04ff"


def _is_lat(ch: str) -> bool:
    return ("a" <= ch <= "z") or ("A" <= ch <= "Z")


def normalize_text(text: str):
    text = unicodedata.normalize("NFKC", str(text))
    has_invisible = bool(_INVISIBLE_RE.search(text))
    text = _INVISIBLE_RE.sub("", text)

    mixed = 0

    def fix_word(m):
        nonlocal mixed
        w = m.group(0)
        n_cyr = sum(_is_cyr(c) for c in w)
        n_lat = sum(_is_lat(c) for c in w)
        if n_cyr and n_lat:
            mixed += 1
            return w.translate(_LAT2CYR) if n_cyr >= n_lat else w.translate(_CYR2LAT)
        return w

    text = _WORD_RE.sub(fix_word, text)
    return text, mixed, has_invisible


def cyrillic_ratio(text: str) -> float:
    n_cyr = n_lat = 0
    for c in text:
        if _is_cyr(c):
            n_cyr += 1
        elif _is_lat(c):
            n_lat += 1
    total = n_cyr + n_lat
    return n_cyr / total if total else 0.0


def lang_tag(cyr: float) -> str:
    if cyr >= 0.7:
        return "ru"
    if cyr <= 0.3:
        return "en"
    return "mixed"


def add_features(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    res = [normalize_text(t) for t in df[text_col]]
    df = df.copy()
    df["clean"] = [r[0] for r in res]
    df["mixed_script"] = [float(r[1] > 0) for r in res]
    df["invisible"] = [float(r[2]) for r in res]
    df["cyr_ratio"] = df["clean"].map(cyrillic_ratio).astype(float)
    df["lang"] = df["cyr_ratio"].map(lang_tag)
    return df
