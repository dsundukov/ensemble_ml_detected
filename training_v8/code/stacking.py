import numpy as np
from sklearn.linear_model import LogisticRegression

FEATURE_NAMES = [
    "logit_tfidf", "logit_rubert", "logit_minilm",
    "cyr_ratio", "cyr_x_rubert", "lat_x_minilm",
    "mixed_script", "invisible",
]


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def meta_features(p_tfidf, p_rubert, p_minilm, cyr, mixed, invisible) -> np.ndarray:
    lt, lr_, lm = logit(p_tfidf), logit(p_rubert), logit(p_minilm)
    cyr = np.asarray(cyr, dtype=float)
    return np.column_stack([
        lt, lr_, lm,
        cyr, cyr * lr_, (1 - cyr) * lm,
        np.asarray(mixed, dtype=float), np.asarray(invisible, dtype=float),
    ])


def train_meta(X, y, seed: int = 42) -> LogisticRegression:
    return LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000,
                              random_state=seed).fit(X, y)


def threshold_at_fpr(scores, y, target_fpr: float) -> float:
    benign = np.asarray(scores)[np.asarray(y) == 0]
    if len(benign) == 0:
        raise ValueError("Нет безобидных примеров для подбора порога")
    return float(np.quantile(benign, 1 - target_fpr, method="higher"))


def threshold_at_recall(scores, y, target_recall: float) -> float:
    attacks = np.asarray(scores)[np.asarray(y) == 1]
    if len(attacks) == 0:
        raise ValueError("Нет атак для подбора порога")
    return float(np.quantile(attacks, 1 - target_recall, method="lower")) - 1e-9


def precision_recall_at(scores, y, threshold: float):
    scores, y = np.asarray(scores), np.asarray(y)
    pred = scores > threshold
    tp = int((pred & (y == 1)).sum())
    precision = tp / max(int(pred.sum()), 1)
    recall = tp / max(int((y == 1).sum()), 1)
    return precision, recall


def cascade_stats(p_tfidf, p_rubert, y, early_benign: float, early_attack: float) -> dict:
    y = np.asarray(y)
    eb = (p_tfidf < early_benign) & (p_rubert < early_benign)
    ea = (p_tfidf > early_attack) & (p_rubert > early_attack)
    return {
        "early_exit_share": float((eb | ea).mean()),
        "attacks_missed_by_early_benign": int((y[eb] == 1).sum()),
        "benign_flagged_by_early_attack": int((y[ea] == 0).sum()),
    }


def cascade_verdict(p_tfidf, p_rubert, score, threshold, gray_low, early_benign, early_attack):
    p_t, p_r, s = (np.asarray(a, dtype=float) for a in (p_tfidf, p_rubert, score))
    v = np.where(s > threshold, 1, np.where(s <= gray_low, 0, -1))
    v = np.where((p_t < early_benign) & (p_r < early_benign), 0, v)
    v = np.where((p_t > early_attack) & (p_r > early_attack), 1, v)
    return v


def oof_meta_scores(X, y, seed: int = 42, n_splits: int = 5) -> np.ndarray:
    from sklearn.model_selection import StratifiedKFold
    y = np.asarray(y)
    n_splits = max(2, min(n_splits, int(np.bincount(y, minlength=2).min())))
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, y):
        oof[te] = train_meta(X[tr], y[tr], seed).predict_proba(X[te])[:, 1]
    return oof
