from device import describe, get_device, setup_cpu

import sys
import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import Config
from data import load_data, split_data
from evaluate import bootstrap_auc, evaluate, three_way_report, wilson
from models_tfidf import build_tfidf_model
from models_transformer import fine_tune, predict_proba
from safetensors_io import save_meta, save_tfidf
from stacking import (FEATURE_NAMES, cascade_stats, cascade_verdict, meta_features, oof_meta_scores,
                      precision_recall_at, threshold_at_fpr, threshold_at_recall, train_meta)

BASE_MODELS = ("tfidf", "rubert_tiny2", "minilm")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def threshold_mask(df, cfg) -> np.ndarray:
    ru = (df["lang"] == "ru").values
    native = ru & ~df["translated"].values
    benign = df["label"].values == 0
    for name, mask in (("нативные русские", native), ("русские с переводами", ru)):
        n = int((mask & benign).sum())
        if n >= cfg.min_ru_benign_for_threshold:
            print(f"  порог подбирается по: {name} ({int(mask.sum())} строк, безобидных {n})")
            return mask
    print(f"[warn] русских безобидных в blend меньше {cfg.min_ru_benign_for_threshold}, "
          f"порог подбирается по всем языкам")
    return np.ones(len(df), bool)


def downsample_attacks(train, cfg):
    if not cfg.max_attack_ratio:
        return train
    src = train["source"].astype(str)
    protected = src.str.startswith(tuple(cfg.protected_sources))
    attack = train["label"].values == 1
    pool = train[attack & ~protected.values]
    n_keep = int(cfg.max_attack_ratio * (train["label"] == 0).sum()) - int((attack & protected.values).sum())
    if len(pool) <= max(n_keep, 0):
        return train
    rng = np.random.default_rng(cfg.seed)
    keep_idx = []
    for _, idx in pool.groupby("source").groups.items():
        n = max(1, round(len(idx) * n_keep / len(pool)))
        keep_idx.extend(rng.choice(np.asarray(idx), size=min(n, len(idx)), replace=False))
    keep = train.loc[keep_idx]
    out = pd.concat([train[~attack | protected.values], keep]).sample(frac=1, random_state=cfg.seed)
    print(f"[data] train прорежен для скорости: {len(train)} -> {len(out)} строк "
          f"(атак {int(attack.sum())} -> {int(out['label'].sum())}, безобидных без изменений)")
    return out.reset_index(drop=True)


def pick_thresholds(scores, y, cfg):
    if cfg.threshold_mode == "recall":
        thr = threshold_at_recall(scores, y, cfg.target_recall)
        gray_low = min(threshold_at_recall(scores, y, cfg.gray_recall), thr)
        mode = f"recall {cfg.target_recall:.0%}"
    else:
        thr = threshold_at_fpr(scores, y, cfg.target_fpr)
        gray_low = min(threshold_at_fpr(scores, y, cfg.gray_fpr), thr)
        mode = f"FPR {cfg.target_fpr:.0%}"
    prec, rec = precision_recall_at(scores, y, thr)
    print(f"  порог ({mode}): {thr:.4f}, серая зона от {gray_low:.4f}")
    print(f"  на выборке подбора: precision={prec:.1%}, recall={rec:.1%}")
    if prec < cfg.min_precision:
        print(f"  [warn] precision ниже цели {cfg.min_precision:.0%}: "
              f"модель не разделяет классы достаточно, нужны данные или снижение target_recall")
    return thr, gray_low


TITLES = {"blend": "blend", "test": "test", "eval": "контрольный набор", "thr": "выборка для порога"}


def print_shares(parts: dict):
    print("[data] состав выборок:")
    for name, d in parts.items():
        n, k = len(d), int(d["label"].sum())
        print(f"  {TITLES.get(name, name):<20} n={n:>6}  атак={k:>6}  безобидных={n - k:>5}  доля атак={k / max(n, 1):.3f}")


def goal_check(r: dict, n_att: int, cfg):
    P, R = cfg.min_precision, getattr(cfg, "min_recall", 0.75)
    print(f"    Цель: precision >= {P:.0%}, recall >= {R:.0%}")
    variants = (("None = нет атаки", r["tp"], r["tp"] + r["fp"]),
                ("None = атака    ", r["tp"] + r["none_attack"],
                 r["tp"] + r["fp"] + r["none_attack"] + r["none_benign"]))
    for title, k, npos in variants:
        prec, rec = k / max(npos, 1), k / max(n_att, 1)
        (lp, hp), (lr, hr) = wilson(k, npos), wilson(k, n_att)
        if prec >= P and rec >= R:
            note = "OK, подтверждено ДИ" if lp >= P and lr >= R else "OK точечно, но ДИ включает значения ниже цели"
        else:
            note = ("не достигнута, статистически значимо" if hp < P or hr < R
                    else "не достигнута, но в пределах ДИ")
        print(f"    {title}: precision={prec:.1%}, recall={rec:.1%} -> {note}")


def report_part(name, part, p, meta, feats, thr, gray_low, cfg, out_dir: Path):
    y = part["label"].values
    s = meta.predict_proba(feats(part, p))[:, 1]
    title = TITLES.get(name, name)
    print(f"\n=== Ансамбль ({title}) === n={len(y)}, атак={int(y.sum())}, доля атак={y.mean():.3f}")
    print(evaluate(y, s, thr, part["lang"].values, cfg.target_fpr).to_string(index=False))

    aucs = bootstrap_auc(y, {"ансамбль": s, "tfidf": p["tfidf"], "rubert_tiny2": p["rubert_tiny2"],
                             "minilm": p["minilm"]}, seed=cfg.seed)
    if aucs:
        print("\n  ROC-AUC [95% бутстреп-ДИ]:")
        for k, (a, lo, hi) in aucs.items():
            if isinstance(k, tuple):
                sig = "значимо" if lo > 0 or hi < 0 else "незначимо"
                print(f"    ансамбль − {k[1]:<13}: {a:+.4f} [{lo:+.4f}; {hi:+.4f}] ({sig})")
            else:
                print(f"    {k:<24}: {a:.4f} [{lo:.4f}; {hi:.4f}]")

    v_meta = np.where(s > thr, 1, np.where(s <= gray_low, 0, -1))
    v_prod = cascade_verdict(p["tfidf"], p["rubert_tiny2"], s, thr, gray_low, cfg.early_benign, cfg.early_attack)
    three_way_report(y, v_meta, "Мета-модель без каскада")
    r = three_way_report(y, v_prod, "Как в проде (с каскадом)")
    goal_check(r, int(y.sum()), cfg)

    cs = cascade_stats(p["tfidf"], p["rubert_tiny2"], y, cfg.early_benign, cfg.early_attack)
    print(f"    каскад: ранний выход {cs['early_exit_share']:.1%}, атак ошибочно отсечено "
          f"{cs['attacks_missed_by_early_benign']}, безобидных ошибочно помечено {cs['benign_flagged_by_early_attack']}")

    if name == "eval":
        verdict = np.select([v_prod == 1, v_prod == 0], ["True", "False"], "None")
        ev = part.assign(verdict=verdict, score=s.round(4))
        ev["outcome"] = np.select(
            [(v_prod == 1) & (y == 1), (v_prod == 1) & (y == 0), (v_prod == 0) & (y == 1), (v_prod == 0) & (y == 0)],
            ["TP", "FP", "FN", "TN"], default="None")
        if "category" in ev.columns:
            table = pd.crosstab(ev["category"], ev["outcome"])
            errors = table.reindex(columns=["FP", "FN", "None"], fill_value=0).sum(axis=1)
            print("\n  Категории с ошибками (как в проде; FP / FN / None):")
            print(table.loc[errors[errors > 0].sort_values(ascending=False).index].to_string())
        cols = [c for c in ("test_id", "category", "text", "label", "verdict", "score", "outcome") if c in ev.columns]
        ev[cols].to_csv(out_dir / "eval_predictions.csv", index=False)
        print(f"  Подробно: {out_dir / 'eval_predictions.csv'}")


def collect_data_files(explicit, data_dir, holdout):
    files = [Path(f) for f in explicit]
    if data_dir:
        found = sorted(Path(data_dir).glob("*.csv"))
        names = {f.name for f in found}
        for f in found:
            if f.name.endswith("_aug.csv") or f"{f.stem}_aug.csv" not in names:
                files.append(f)
    hold = {Path(h).resolve() for h in holdout}
    hold_names = {Path(h).name for h in holdout}
    result, seen = [], set()
    for f in files:
        r = f.resolve()
        if r in hold or f.name in hold_names:
            continue
        if r not in seen:
            seen.add(r)
            result.append(f)
    if not result:
        sys.exit("Нет обучающих файлов: укажите --data или --data-dir")
    print("[data] файлы для обучения:")
    for f in result:
        try:
            d = pd.read_csv(f, usecols=["label"])
            print(f"  {f}  строк={len(d)}  атак={int(d['label'].sum())}")
        except Exception as e:
            sys.exit(f"{f}: не удалось прочитать ({e})")
    if hold_names:
        print(f"[data] исключены из обучения (проверочные выборки): {', '.join(sorted(hold_names))}")
    return [str(f) for f in result]


def fine_tune_checked(tcfg, train, model_dir, cfg, device):
    import dataclasses
    min_auc = getattr(cfg, "min_val_auc", 0.80)
    retries = getattr(cfg, "max_retries", 2)
    cur = tcfg
    for attempt in range(retries + 1):
        seed = cfg.seed + attempt
        model, tok = fine_tune(cur, train["clean"].tolist(), train["label"].tolist(),
                               str(model_dir), seed, cfg.val_size, device,
                               groups=train["group"].tolist())
        auc = getattr(model, "best_val_auc", 1.0)
        if auc >= min_auc:
            if attempt:
                print(f"  {tcfg.name}: обучилась с попытки {attempt + 1} (lr={cur.lr:g}, seed={seed}, "
                      f"ROC-AUC {auc:.4f})")
            return model, tok
        print(f"[warn] {tcfg.name}: ROC-AUC на валидации {auc:.4f} < {min_auc} - модель не обучилась "
              f"(попытка {attempt + 1}/{retries + 1}, lr={cur.lr:g}, seed={seed})")
        del model
        cur = dataclasses.replace(cur, lr=cur.lr / 2)
    raise RuntimeError(f"{tcfg.name} не обучилась за {retries + 1} попыток (ROC-AUC < {min_auc}). "
                       f"Уменьшите lr в config.py или уберите модель из ансамбля.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="*", default=[], help="CSV с колонками text,label[,group,source]")
    ap.add_argument("--data-dir", default=None,
                    help="взять для обучения ВСЕ CSV из папки (публичные, PromptSC, сгенерированные), кроме "
                         "--eval-data и --threshold-data; если есть X.csv и X_aug.csv, берётся X_aug.csv")
    ap.add_argument("--eval-data", nargs="*", default=[],
                    help="Контрольный набор: только оценка, в обучении и подборе порога не участвует")
    ap.add_argument("--threshold-data", nargs="*", default=[],
                    help="Доменная выборка для подбора порогов (вместо blend); в обучении не участвует")
    ap.add_argument("--output", default=None)
    ap.add_argument("--seed", type=int, default=None, help="переопределить seed из config.py")
    args = ap.parse_args()

    cfg = Config()
    if args.output:
        cfg.output_dir = args.output
    if args.seed is not None:
        cfg.seed = args.seed
    out = Path(cfg.output_dir)
    (out / "models").mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    if device == "cpu":
        setup_cpu(cfg.num_threads)
    print(f"[device] обучение на: {describe(device)} | seed={cfg.seed}")

    data_files = collect_data_files(args.data, args.data_dir, args.eval_data + args.threshold_data)
    df = load_data(data_files)
    train, blend, test = split_data(df, cfg.n_folds, cfg.seed)
    train = downsample_attacks(train, cfg)

    extra = {}
    if args.eval_data:
        extra["eval"] = load_data(args.eval_data)
    if args.threshold_data:
        extra["thr"] = load_data(args.threshold_data)
    if "eval" in extra and "thr" in extra:
        dup = extra["thr"]["clean"].isin(set(extra["eval"]["clean"]))
        if dup.any():
            print(f"[warn] {int(dup.sum())} текстов выборки для порога есть в контрольной - исключены")
            extra["thr"] = extra["thr"][~dup].reset_index(drop=True)
    if extra:
        held = set().union(*(set(d["clean"]) for d in extra.values()))
        n_before = len(train) + len(blend) + len(test)
        train, blend, test = (d[~d["clean"].isin(held)].reset_index(drop=True) for d in (train, blend, test))
        print(f"[data] исключено пересечений с отложенными выборками: {n_before - len(train) - len(blend) - len(test)}")

    parts = {"blend": blend, "test": test, **extra}
    print_shares({"train": train, **parts})
    preds = {name: {} for name in parts}

    print("\n[1/4] TF-IDF + LogReg")
    tfidf = build_tfidf_model(cfg.seed).fit(train["clean"], train["label"])
    save_tfidf(tfidf, out / "models" / "tfidf")
    for name, part in parts.items():
        preds[name]["tfidf"] = tfidf.predict_proba(part["clean"])[:, 1]

    for i, tcfg in enumerate(cfg.transformers, start=2):
        print(f"\n[{i}/4] {tcfg.hf_id}")
        model_dir = out / "models" / tcfg.name
        model, tok = fine_tune_checked(tcfg, train, model_dir, cfg, device)
        assert (model_dir / "model.safetensors").exists(), "веса не сохранились в safetensors"
        for name, part in parts.items():
            preds[name][tcfg.name] = predict_proba(model, tok, part["clean"].tolist(),
                                                   tcfg.max_length, cfg.stride)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()

    np.savez(out / "predictions.npz",
             **{f"{s}_{k}": v for s, d in preds.items() for k, v in d.items()})

    print("\n[4/4] Мета-модель (стекинг)")

    def feats(part, p):
        return meta_features(p["tfidf"], p["rubert_tiny2"], p["minilm"],
                             part["cyr_ratio"].values, part["mixed_script"].values,
                             part["invisible"].values)

    Xb, yb = feats(blend, preds["blend"]), blend["label"].values
    meta = train_meta(Xb, yb, cfg.seed)
    save_meta(meta, FEATURE_NAMES, out / "models" / "meta")
    for fname, coef in zip(FEATURE_NAMES, meta.coef_[0]):
        print(f"  {fname:>14}: {coef:+.3f}")

    if "thr" in parts:
        td = parts["thr"]
        s_thr = meta.predict_proba(feats(td, preds["thr"]))[:, 1]
        print(f"  пороги подбираются по выборке --threshold-data: n={len(td)}, атак={int(td['label'].sum())}")
        thr, gray_low = pick_thresholds(s_thr, td["label"].values, cfg)
        source = "threshold_data"
    else:
        mask = threshold_mask(blend, cfg)
        oof = oof_meta_scores(Xb, yb, cfg.seed)
        print("  пороги подбираются по out-of-fold оценкам мета-модели на blend")
        thr, gray_low = pick_thresholds(oof[mask], yb[mask], cfg)
        source = "blend_oof"

    mask_b = threshold_mask(blend, cfg) if source == "threshold_data" else mask
    yt, langs = test["label"].values, test["lang"].values
    for name in BASE_MODELS:
        t_ = threshold_at_fpr(preds["blend"][name][mask_b], yb[mask_b], cfg.target_fpr)
        print(f"\n=== {name} (test) ===")
        print(evaluate(yt, preds["test"][name], t_, langs, cfg.target_fpr).to_string(index=False))

    for name in ("test", "thr", "eval"):
        if name in parts:
            report_part(name, parts[name], preds[name], meta, feats, thr, gray_low, cfg, out)

    ens_cfg = {
        "threshold": thr,
        "gray_low": gray_low,
        "threshold_source": source,
        "threshold_mode": cfg.threshold_mode,
        "target_recall": cfg.target_recall,
        "target_fpr": cfg.target_fpr,
        "seed": cfg.seed,
        "stride": cfg.stride,
        "early_benign": cfg.early_benign,
        "early_attack": cfg.early_attack,
        "feature_names": FEATURE_NAMES,
        "transformers": [asdict(t) for t in cfg.transformers],
        "prompt_guard_id": cfg.prompt_guard_id,
        "prompt_guard_low": cfg.prompt_guard_low,
        "prompt_guard_high": cfg.prompt_guard_high,
    }
    (out / "ensemble_config.json").write_text(
        json.dumps(ens_cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nГотово. Артефакты сохранены в {out}/")


if __name__ == "__main__":
    main()
