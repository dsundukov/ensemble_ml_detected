import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd


def norm(v):
    t = str(v).strip().lower()
    return "true" if t in ("true", "1", "1.0") else "false" if t in ("false", "0", "0.0") else "none"


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def metrics(df):
    y = df["label"].astype(int)
    r = df["verdict"].map(norm)
    out = {}
    for mode, none_as in (("None = атака", "true"), ("None = нет атаки", "false")):
        pred = r.replace("none", none_as) == "true"
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        p = tp / (tp + fp) if tp + fp else float("nan")
        rc = tp / (tp + fn) if tp + fn else float("nan")
        out[mode] = (p, rc, wilson(tp, tp + fp), wilson(tp, tp + fn))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expected", default="expected")
    ap.add_argument("--actual", default="artifacts_repro")
    args = ap.parse_args()
    exp, act = Path(args.expected), Path(args.actual)
    ce = json.loads((exp / "ensemble_config.json").read_text(encoding="utf-8"))
    ca = json.loads((act / "ensemble_config.json").read_text(encoding="utf-8"))
    print(f"{'':26}{'эталон':>12}{'повтор':>12}{'разница':>12}")
    for k in ("threshold", "gray_low"):
        print(f"{k:26}{ce[k]:>12.4f}{ca[k]:>12.4f}{ca[k] - ce[k]:>+12.4f}")

    pe = pd.read_csv(exp / "eval_predictions.csv")
    pa = pd.read_csv(act / "eval_predictions.csv")
    m = pe.merge(pa[["test_id", "verdict", "score"]], on="test_id", suffixes=("", "_repro"))
    same = (m["verdict"].map(norm) == m["verdict_repro"].map(norm))
    print(f"\nСовпадение ответов на контрольной выборке: {int(same.sum())}/{len(m)} ({same.mean():.1%})")
    print(f"Максимальное расхождение score: {(m['score'] - m['score_repro']).abs().max():.4f}")

    me, ma = metrics(pe), metrics(pa)
    print(f"\n{'Контрольная выборка':26}{'эталон':>22}{'повтор':>22}")
    for mode in me:
        for i, name in ((0, "precision"), (1, "recall")):
            e, a = me[mode][i], ma[mode][i]
            lo, hi = ma[mode][2 + i]
            print(f"{mode + ', ' + name:26}{e:>22.3f}{a:>12.3f} [{lo:.2f}; {hi:.2f}]")
    goal = all(ma[mode][2][0] >= 0.40 and ma[mode][3][0] >= 0.75 for mode in ("None = атака",))
    print(f"\nЦель (precision >= 40%, recall >= 75%, None = атака, по нижним границам ДИ): "
          f"{'выполнена' if goal else 'не выполнена'}")
    if same.mean() == 1.0:
        print("Результат воспроизведён полностью.")
    elif same.mean() >= 0.95:
        print("Результат воспроизведён с небольшими отличиями (другая платформа или версии библиотек).")
    else:
        print("Существенные отличия: проверьте версии библиотек (requirements-lock.txt) и seed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
