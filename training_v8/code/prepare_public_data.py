import hashlib
from pathlib import Path

import pandas as pd
from datasets import load_dataset


def _md5(t: str) -> str:
    return hashlib.md5(str(t).encode("utf-8")).hexdigest()


def rows_deepset(r):
    yield r["text"], int(r["label"]), _md5(r["text"])


def rows_jackhhao(r):
    yield r["prompt"], int(r["type"] == "jailbreak"), _md5(r["prompt"])


def rows_lakera(r):
    yield r["text"], 1, _md5(r["text"])


def rows_dmtrdr(r):
    group = _md5(r.get("prompt_ru") or r.get("prompt_en"))
    for col in ("prompt_ru", "prompt_en"):
        if r.get(col):
            yield r[col], 1, group


SOURCES = {
    "deepset/prompt-injections": rows_deepset,
    "jackhhao/jailbreak-classification": rows_jackhhao,
    "Lakera/gandalf_ignore_instructions": rows_lakera,
    "dmtrdr/russian_prompt_injections": rows_dmtrdr,
}


def main():
    rows = []
    for name, fn in SOURCES.items():
        try:
            ds = load_dataset(name)
            n0 = len(rows)
            for split in ds:
                for r in ds[split]:
                    for text, label, group in fn(r):
                        rows.append({"text": text, "label": label, "group": group, "source": name})
            print(f"[ok]   {name}: {len(rows) - n0} строк")
        except Exception as e:
            print(f"[skip] {name}: {e}")

    df = pd.DataFrame(rows).drop_duplicates("text")
    Path("data").mkdir(exist_ok=True)
    df.to_csv("data/public.csv", index=False)
    print(f"\nСохранено {len(df)} строк в data/public.csv")
    print(df.groupby(["source", "label"]).size().to_string())


if __name__ == "__main__":
    main()
