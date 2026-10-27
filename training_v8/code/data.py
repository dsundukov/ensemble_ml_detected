import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from preprocessing import add_features


def _md5(t: str) -> str:
    return hashlib.md5(t.encode("utf-8")).hexdigest()


_NON_LETTERS = re.compile(r"[\W\d_]+")


def _skeleton(text: str) -> str:
    return " ".join(_NON_LETTERS.sub(" ", text.lower()).split())


def merge_near_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    parent = {}

    def find(x):
        root = x
        while parent.get(root, root) != root:
            root = parent[root]
        while parent.get(x, x) != root:
            parent[x], x = root, parent[x]
        return root

    for _, groups in df.groupby(df["clean"].map(_skeleton))["group"]:
        g = groups.unique()
        r0 = find(g[0])
        for other in g[1:]:
            ro = find(other)
            if ro != r0:
                parent[ro] = r0
    before = df["group"].nunique()
    df = df.assign(group=df["group"].map(find))
    sizes = df["group"].value_counts()
    print(f"[data] почти-дубликаты: групп {before} -> {df['group'].nunique()} "
          f"(объединено {before - df['group'].nunique()}); крупнейшая группа {sizes.iloc[0]} строк "
          f"({sizes.iloc[0] / len(df):.1%})")
    if sizes.iloc[0] > 0.05 * len(df):
        print("[warn] крупнейшая группа больше 5% данных: разбиение на выборки может стать несбалансированным")
    return df


def load_data(paths, merge_near_dups: bool = True) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        missing = {"text", "label"} - set(df.columns)
        if missing:
            raise ValueError(f"{p}: нет колонок {missing}")
        if "source" not in df.columns:
            df["source"] = Path(p).stem
        frames.append(df)

    df = pd.concat(frames, ignore_index=True).dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str)
    df = df[df["text"].str.strip().str.len() > 0]
    df["label"] = df["label"].astype(int)
    if not set(df["label"].unique()) <= {0, 1}:
        raise ValueError("label должен быть 0 (безобидный) или 1 (атака)")

    for col in ("augmented", "translated"):
        df[col] = (df[col].map(lambda v: str(v).strip().lower() == "true")
                   if col in df.columns else False)

    df = add_features(df)

    n_labels = df.groupby("clean")["label"].nunique()
    conflicts = n_labels[n_labels > 1].index
    if len(conflicts):
        print(f"[data] удалено {len(conflicts)} текстов с противоречивой разметкой")
    df = df[~df["clean"].isin(conflicts)].drop_duplicates(subset="clean").reset_index(drop=True)

    if "group" not in df.columns:
        df["group"] = np.nan
    no_group = df["group"].isna()
    df.loc[no_group, "group"] = df.loc[no_group, "clean"].map(_md5)
    df["group"] = df["group"].astype(str)
    if merge_near_dups:
        df = merge_near_duplicates(df)

    print("[data] распределение (язык x метка):")
    print(pd.crosstab(df["lang"], df["label"]).to_string())
    return df


def split_data(df: pd.DataFrame, n_folds: int, seed: int):
    strat = df["label"].astype(str) + "_" + df["lang"]
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = np.zeros(len(df), dtype=int)
    for i, (_, idx) in enumerate(sgkf.split(df, strat, groups=df["group"])):
        folds[idx] = i
    aug = df["augmented"].values
    test = df[(folds == 0) & ~aug].reset_index(drop=True)
    blend = df[(folds == 1) & ~aug].reset_index(drop=True)
    train = df[folds >= 2].reset_index(drop=True)
    return train, blend, test
