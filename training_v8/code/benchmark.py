import argparse
import json
import platform
import time

import numpy as np
import pandas as pd

from ensemble_model import PromptAttackEnsemble


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ensemble")
    ap.add_argument("--data", required=True, help="CSV с колонкой text")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--repeat", type=int, default=1, help="сколько раз прогнать набор")
    ap.add_argument("--limit-ms", type=float, default=200.0)
    args = ap.parse_args()

    texts = pd.read_csv(args.data)["text"].astype(str).tolist()
    model = PromptAttackEnsemble(args.model, args.threads)
    for t in texts[:5]:
        model.predict(t)

    times, stages = [], []
    for _ in range(args.repeat):
        for t in texts:
            t0 = time.perf_counter()
            r = model.predict(t)
            times.append((time.perf_counter() - t0) * 1000)
            stages.append(r["stage"])
    ms = np.array(times)
    lengths = np.array([len(t) for t in texts] * args.repeat)

    report = {
        "cpu": platform.processor() or platform.machine(),
        "threads": args.threads or "все",
        "texts": len(texts), "measurements": len(ms),
        "mean_ms": round(float(ms.mean()), 1),
        "p50_ms": round(float(np.percentile(ms, 50)), 1),
        "p95_ms": round(float(np.percentile(ms, 95)), 1),
        "p99_ms": round(float(np.percentile(ms, 99)), 1),
        "max_ms": round(float(ms.max()), 1),
        f"share_over_{int(args.limit_ms)}ms": round(float((ms > args.limit_ms).mean()), 4),
        "p95_ms_long_texts_top10pct": round(float(np.percentile(ms[lengths >= np.percentile(lengths, 90)], 95)), 1),
        "stages": pd.Series(stages).value_counts(normalize=True).round(3).to_dict(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    ok = report["p95_ms"] <= args.limit_ms
    print(f"\np95 = {report['p95_ms']} мс -> {'требование выполнено' if ok else 'ТРЕБОВАНИЕ НЕ ВЫПОЛНЕНО'} "
          f"(лимит {args.limit_ms:.0f} мс)")


if __name__ == "__main__":
    main()
