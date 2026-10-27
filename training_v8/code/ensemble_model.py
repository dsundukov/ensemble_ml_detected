import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Optional

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from safetensors.numpy import load_file
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize as sk_normalize
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


_INVISIBLE_RE = re.compile("[­​-‏‪-‮⁠-⁤﻿]")
_WORD_RE = re.compile(r"\w+")
_LAT = "aceopxyACEHKMOPTXB"
_CYR = "асеорхуАСЕНКМОРТХВ"
_LAT2CYR = str.maketrans(_LAT, _CYR)
_CYR2LAT = str.maketrans(_CYR, _LAT)


def _is_cyr(ch):
    return "Ѐ" <= ch <= "ӿ"


def _is_lat(ch):
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

    return _WORD_RE.sub(fix_word, text), mixed > 0, has_invisible


def cyrillic_ratio(text: str) -> float:
    n_cyr = sum(_is_cyr(c) for c in text)
    n_lat = sum(_is_lat(c) for c in text)
    return n_cyr / (n_cyr + n_lat) if (n_cyr + n_lat) else 0.0


def lang_tag(cyr: float) -> str:
    return "ru" if cyr >= 0.7 else "en" if cyr <= 0.3 else "mixed"


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _logit(p):
    p = np.clip(float(p), 1e-5, 1 - 1e-5)
    return float(np.log(p / (1 - p)))


class _Tfidf:

    def __init__(self, root: Path, vectorizers, w):
        self.parts = []
        for v in vectorizers:
            vocab = json.loads((root / f"vocab_{v['name']}.json").read_text(encoding="utf-8"))
            counter = CountVectorizer(analyzer=v["analyzer"], ngram_range=tuple(v["ngram_range"]),
                                      lowercase=v["lowercase"], token_pattern=v["token_pattern"],
                                      vocabulary=vocab, dtype=np.float64)
            self.parts.append((counter, w[f"tfidf.{v['name']}.idf"].astype(np.float64),
                               v["sublinear_tf"], v["norm"]))
        self.coef = w["tfidf.clf.coef"].astype(np.float64)
        self.intercept = float(w["tfidf.clf.intercept"][0])

    def __call__(self, text: str) -> float:
        z, offset = self.intercept, 0
        for counter, idf, sublinear, norm in self.parts:
            X = counter.transform([text]).tocsr()
            if sublinear:
                X.data = np.log(X.data) + 1.0
            X = X.multiply(idf).tocsr()
            if norm:
                X = sk_normalize(X, norm=norm)
            z += float((X @ self.coef[offset:offset + len(idf)])[0])
            offset += len(idf)
        return float(_sigmoid(z))


_LEGACY_SUFFIXES = (("LayerNorm.gamma", "LayerNorm.weight"), ("LayerNorm.beta", "LayerNorm.bias"))


def _modern_key(key: str) -> str:
    for old, new in _LEGACY_SUFFIXES:
        if key.endswith(old):
            return key[: -len(old)] + new
    return key


class _Transformer:

    def __init__(self, model_dir: Path, prefix: str, w, max_length: int, stride: int):
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        state = {_modern_key(k[len(prefix) + 1:]): v
                 for k, v in w.items() if k.startswith(prefix + ".")}
        self.model = self._build(model_dir, prefix, state)
        self.model.eval()
        self.max_length, self.stride = max_length, min(stride, max_length // 4)

    @staticmethod
    def _build(model_dir: Path, prefix: str, state: dict):
        config = AutoConfig.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_config(config)
        missing, unexpected = model.load_state_dict(
            {k: torch.from_numpy(v.copy()) for k, v in state.items()}, strict=False)
        missing = [k for k in missing if not k.endswith("position_ids")]
        if not missing and not unexpected:
            return model

        import tempfile
        from safetensors.numpy import save_file
        with tempfile.TemporaryDirectory() as tmp:
            config.save_pretrained(tmp)
            save_file({k: np.ascontiguousarray(v) for k, v in state.items()}, f"{tmp}/model.safetensors")
            model = AutoModelForSequenceClassification.from_pretrained(tmp)
        loaded = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()
                  if not k.endswith("position_ids")}
        from_file = {_modern_key(k): v for k, v in state.items()}
        bad = [k for k, v in loaded.items()
               if k not in from_file or v.shape != from_file[k].shape
               or not np.allclose(v, from_file[k], atol=1e-6)]
        if bad:
            raise ValueError(f"{prefix}: веса не совпали с файлом: {bad[:5]}. "
                             f"Пересоберите ансамбль: python pack_ensemble.py --artifacts artifacts --out ensemble")
        return model

    @torch.inference_mode()
    def __call__(self, text: str) -> float:
        enc = self.tok([text], truncation=True, max_length=self.max_length, stride=self.stride,
                       return_overflowing_tokens=True, padding=True, return_tensors="pt")
        enc.pop("overflow_to_sample_mapping", None)
        logits = self.model(**enc).logits
        return float(torch.softmax(logits.float(), -1)[:, 1].max())


def _resolve_model_dir(path) -> Path:
    root = Path(path).expanduser()
    if (root / "ensemble_config.json").exists() and (root / "model.safetensors").exists():
        return root
    nested = root / "ensemble"
    if (nested / "ensemble_config.json").exists() and (nested / "model.safetensors").exists():
        return nested
    hint = [f"Не найден ансамбль в папке: {root.resolve()}"]
    if not root.exists():
        hint.append("Папки нет. Выполните упаковку и скопируйте результат сюда:")
    else:
        found = sorted(p.name for p in root.iterdir())
        hint.append(f"Папка есть, но в ней не хватает файлов. Сейчас там: {found or 'пусто'}")
        hint.append("Должны быть: model.safetensors, ensemble_config.json, vocab_*.json, rubert_tiny2/, minilm/.")
    hint.append("  1) cd <папка обучения> && python pack_ensemble.py --artifacts artifacts --out ensemble")
    hint.append(f"  2) cp -R <папка обучения>/ensemble {root.resolve().parent}/")
    raise FileNotFoundError("\n".join(hint))


class PromptAttackEnsemble:

    def __init__(self, path: str = "ensemble", num_threads: Optional[int] = None):
        root = _resolve_model_dir(path)
        if num_threads:
            torch.set_num_threads(num_threads)
        self.cfg = json.loads((root / "ensemble_config.json").read_text(encoding="utf-8"))
        w = load_file(str(root / "model.safetensors"))

        self.tfidf = _Tfidf(root, self.cfg["tfidf_vectorizers"], w)
        self.transformers = {
            t["name"]: _Transformer(root / t["name"], t["name"], w, t["max_length"], self.cfg["stride"])
            for t in self.cfg["transformers"]
        }
        self.meta_coef = w["meta.coef"].astype(np.float64)
        self.meta_intercept = float(w["meta.intercept"][0])

    def __call__(self, text: str) -> Optional[bool]:
        return self.predict(text)["result"]

    def predict(self, text: str) -> dict:
        if not isinstance(text, str):
            raise TypeError(f"Ожидается строка, получено {type(text).__name__}")
        if not text.strip():
            return {"result": None, "score": None, "stage": "empty", "lang": None}

        c = self.cfg
        clean, mixed, invisible = normalize_text(text)
        cyr = cyrillic_ratio(clean)
        lang = lang_tag(cyr)

        p_t = self.tfidf(clean)
        p_r = self.transformers["rubert_tiny2"](clean)
        if p_t < c["early_benign"] and p_r < c["early_benign"]:
            return {"result": False, "score": round(max(p_t, p_r), 4), "stage": "early_benign", "lang": lang}
        if p_t > c["early_attack"] and p_r > c["early_attack"]:
            return {"result": True, "score": round(min(p_t, p_r), 4), "stage": "early_attack", "lang": lang}

        p_m = self.transformers["minilm"](clean)
        lt, lr_, lm = _logit(p_t), _logit(p_r), _logit(p_m)
        x = np.array([lt, lr_, lm, cyr, cyr * lr_, (1 - cyr) * lm, float(mixed), float(invisible)])
        score = float(_sigmoid(x @ self.meta_coef + self.meta_intercept))

        if score > c["threshold"]:
            result = True
        elif score <= c["gray_low"]:
            result = False
        else:
            result = None
        return {
            "result": result,
            "score": round(score, 4),
            "stage": "meta" if result is not None else "gray_zone",
            "lang": lang,
            "components": {"tfidf": round(p_t, 4), "rubert_tiny2": round(p_r, 4), "minilm": round(p_m, 4)},
        }


if __name__ == "__main__":
    import argparse
    import time

    ap = argparse.ArgumentParser(description="Проверка текста на промпт-атаку")
    ap.add_argument("texts", nargs="+")
    ap.add_argument("--model", default="ensemble", help="папка с model.safetensors")
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()

    t0 = time.perf_counter()
    model = PromptAttackEnsemble(args.model, args.threads)
    print(f"Ансамбль загружен за {time.perf_counter() - t0:.1f} с")
    for text in args.texts:
        t0 = time.perf_counter()
        res = model.predict(text)
        ms = (time.perf_counter() - t0) * 1000
        print(json.dumps({"text": text[:80], **res, "ms": round(ms, 1)}, ensure_ascii=False))
