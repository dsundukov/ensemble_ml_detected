import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

_VEC_PARAMS = ("analyzer", "ngram_range", "lowercase", "token_pattern")


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def save_tfidf(pipeline, out_dir) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    union, clf = pipeline.named_steps["features"], pipeline.named_steps["clf"]

    tensors, cfg = {}, {"vectorizers": []}
    for name, vec in union.transformer_list:
        params = {p: getattr(vec, p) for p in _VEC_PARAMS}
        params["ngram_range"] = list(params["ngram_range"])
        params["sublinear_tf"] = bool(vec.sublinear_tf)
        params["norm"] = vec.norm
        cfg["vectorizers"].append({"name": name, **params})
        tensors[f"{name}.idf"] = vec.idf_.astype(np.float32)
        vocab = {k: int(v) for k, v in vec.vocabulary_.items()}
        (out_dir / f"vocab_{name}.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")

    tensors["clf.coef"] = clf.coef_[0].astype(np.float32)
    tensors["clf.intercept"] = clf.intercept_.astype(np.float32)
    save_file(tensors, str(out_dir / "model.safetensors"), metadata={"type": "tfidf_logreg"})
    (out_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


class SafeTfidf:

    def __init__(self, model_dir):
        model_dir = Path(model_dir)
        cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        w = load_file(str(model_dir / "model.safetensors"))
        self.parts = []
        for v in cfg["vectorizers"]:
            vocab = json.loads((model_dir / f"vocab_{v['name']}.json").read_text(encoding="utf-8"))
            counter = CountVectorizer(analyzer=v["analyzer"], ngram_range=tuple(v["ngram_range"]),
                                      lowercase=v["lowercase"], token_pattern=v["token_pattern"],
                                      vocabulary=vocab, dtype=np.float64)
            self.parts.append((counter, w[f"{v['name']}.idf"].astype(np.float64),
                               v["sublinear_tf"], v["norm"]))
        self.coef = w["clf.coef"].astype(np.float64)
        self.intercept = float(w["clf.intercept"][0])

    def predict_proba(self, texts) -> np.ndarray:
        z = np.full(len(texts), self.intercept)
        offset = 0
        for counter, idf, sublinear, norm in self.parts:
            X = counter.transform(texts).tocsr()
            if sublinear:
                X.data = np.log(X.data) + 1.0
            X = X.multiply(idf).tocsr()
            if norm:
                X = normalize(X, norm=norm)
            z += X @ self.coef[offset:offset + len(idf)]
            offset += len(idf)
        p = _sigmoid(z)
        return np.column_stack([1 - p, p])


def save_meta(meta, feature_names, out_dir) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(
        {"coef": meta.coef_[0].astype(np.float32), "intercept": meta.intercept_.astype(np.float32)},
        str(out_dir / "model.safetensors"),
        metadata={"type": "logreg_meta", "feature_names": json.dumps(feature_names)},
    )


class SafeMeta:
    def __init__(self, model_dir):
        w = load_file(str(Path(model_dir) / "model.safetensors"))
        self.coef = w["coef"].astype(np.float64)
        self.intercept = float(w["intercept"][0])

    def predict_proba(self, X) -> np.ndarray:
        p = _sigmoid(np.asarray(X, dtype=np.float64) @ self.coef + self.intercept)
        return np.column_stack([1 - p, p])
