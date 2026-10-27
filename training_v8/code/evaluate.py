import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def tpr_at_fpr(y, scores, target_fpr: float) -> float:
    fpr, tpr, _ = roc_curve(y, scores)
    return float(np.interp(target_fpr, fpr, tpr))


def evaluate(y, scores, threshold: float, langs, target_fpr: float) -> pd.DataFrame:
    y, scores, langs = np.asarray(y), np.asarray(scores), np.asarray(langs)
    rows = []
    for group in ("all", "ru", "en", "mixed"):
        m = np.ones(len(y), bool) if group == "all" else (langs == group)
        if m.sum() == 0:
            continue
        yy, ss = y[m], scores[m]
        pred = ss > threshold
        row = {"group": group, "n": int(m.sum()), "attacks": int(yy.sum())}
        if len(np.unique(yy)) == 2:
            row["roc_auc"] = roc_auc_score(yy, ss)
            row["pr_auc"] = average_precision_score(yy, ss)
            row[f"tpr@fpr{target_fpr:.0%}"] = tpr_at_fpr(yy, ss, target_fpr)
        tp = int((pred & (yy == 1)).sum())
        fp = int((pred & (yy == 0)).sum())
        fn = int((~pred & (yy == 1)).sum())
        tn = int((~pred & (yy == 0)).sum())
        row["precision"] = tp / max(tp + fp, 1)
        row["recall"] = tp / max(tp + fn, 1)
        row["fpr"] = fp / max(fp + tn, 1)
        rows.append(row)
    return pd.DataFrame(rows).round(4)


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def fmt_rate(k: int, n: int) -> str:
    if n == 0:
        return "н/д (n = 0)"
    lo, hi = wilson(k, n)
    return f"{k / n:.3f} [{lo:.2f}; {hi:.2f}] ({k}/{n})"


def bootstrap_auc(y, score_sets: dict, n_boot: int = 500, seed: int = 0) -> dict:
    y = np.asarray(y)
    names = list(score_sets)
    if len(np.unique(y)) < 2:
        return {}
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    boots = {n: [] for n in names}
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        for n in names:
            boots[n].append(roc_auc_score(y[idx], np.asarray(score_sets[n])[idx]))
    out = {}
    for n in names:
        b = np.array(boots[n])
        out[n] = (roc_auc_score(y, score_sets[n]), *np.percentile(b, [2.5, 97.5]))
    ref = names[0]
    for n in names[1:]:
        d = np.array(boots[ref]) - np.array(boots[n])
        out[("Δ", n)] = (out[ref][0] - out[n][0], *np.percentile(d, [2.5, 97.5]))
    return out


def three_way_report(y, verdict, title: str) -> dict:
    y, v = np.asarray(y), np.asarray(verdict)
    att = int((y == 1).sum())
    tp, fp = int(((v == 1) & (y == 1)).sum()), int(((v == 1) & (y == 0)).sum())
    fn, tn = int(((v == 0) & (y == 1)).sum()), int(((v == 0) & (y == 0)).sum())
    na, nb = int(((v == -1) & (y == 1)).sum()), int(((v == -1) & (y == 0)).sum())
    print(f"\n  {title}: TP={tp} FP={fp} FN={fn} TN={tn} None={na + nb} (атак среди None: {na})")
    print(f"    None = нет атаки : precision {fmt_rate(tp, tp + fp)} | recall {fmt_rate(tp, att)} "
          f"| FPR {fmt_rate(fp, fp + tn + nb)}")
    print(f"    None = атака     : precision {fmt_rate(tp + na, tp + fp + na + nb)} | "
          f"recall {fmt_rate(tp + na, att)} | FPR {fmt_rate(fp + nb, fp + tn + nb)}")
    print(f"    доля None        : {fmt_rate(na + nb, len(y))}")
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "none_attack": na, "none_benign": nb}
