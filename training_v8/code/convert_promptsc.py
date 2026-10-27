import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

TEXT_COL = "Входные данные (фрагмент трейса)"
LABELS = {"Injection": 1, "SAFE": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx")
    ap.add_argument("--eval-share", type=float, default=0.5,
                    help="Доля в контрольный набор (0 - всё в обучение)")
    ap.add_argument("--threshold-share", type=float, default=0.0,
                    help="Доля выборки для подбора порогов (0 - не выделять)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw = pd.read_excel(args.xlsx, sheet_name="Данные")
    unknown = set(raw["Эталон"].dropna()) - set(LABELS)
    if unknown:
        raise ValueError(f"Неизвестные значения в колонке «Эталон»: {unknown}")

    df = pd.DataFrame({
        "text": raw[TEXT_COL].astype(str).str.strip(),
        "label": raw["Эталон"].map(LABELS),
        "group": raw["test_id"].astype(str),
        "source": "PromptSC_V5",
        "test_id": raw["test_id"],
        "code": raw["Код"],
        "category": raw["Категория"],
        "unanimous": raw["Единогласно"].eq("да"),
    }).dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)

    print(f"Всего: {len(df)} | атак: {df.label.sum()} | безобидных: {(df.label == 0).sum()} "
          f"| спорных (не единогласно): {(~df.unanimous).sum()}")

    Path("data").mkdir(exist_ok=True)
    if args.eval_share <= 0:
        df.to_csv("data/ru_promptsc_train.csv", index=False)
        print("Сохранено: data/ru_promptsc_train.csv (контрольного набора нет)")
        return

    if args.eval_share + args.threshold_share >= 1:
        raise ValueError("--eval-share + --threshold-share должны быть меньше 1")

    def split(frame, share):
        for strat in (frame["code"], frame["label"]):
            try:
                return train_test_split(frame, test_size=share, stratify=strat, random_state=args.seed)
            except ValueError:
                continue
        return train_test_split(frame, test_size=share, random_state=args.seed)

    rest, ev = split(df, args.eval_share)
    parts = {"eval": ev}
    if args.threshold_share > 0:
        train, thr = split(rest, args.threshold_share / (1 - args.eval_share))
        parts["threshold"] = thr
        thr.to_csv("data/ru_promptsc_threshold.csv", index=False)
    else:
        train = rest
    parts["train"] = train
    train.to_csv("data/ru_promptsc_train.csv", index=False)
    ev.to_csv("data/ru_promptsc_eval.csv", index=False)
    for name, part in parts.items():
        print(f"  {name:<9}: {len(part)} строк (атак {part.label.sum()}, безобидных {(part.label == 0).sum()})")


if __name__ == "__main__":
    main()
