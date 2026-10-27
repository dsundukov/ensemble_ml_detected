import argparse
import random
import re
from pathlib import Path

import pandas as pd

PREFIXES = ["Здравствуйте! ", "Добрый день. ", "Привет, ", "Подскажите, пожалуйста: ",
            "Слушай, ", "Вопрос: ", "Срочно! ", "Уважаемый ассистент, "]
SUFFIXES = [" Спасибо.", " Заранее спасибо!", " Очень срочно.", " Жду ответа.",
            " Пожалуйста.", " Ответь быстро.", " :)"]
SURNAMES = ["Иванова", "Петрова", "Сидорова", "Кузнецова", "Смирнова", "Волкова", "Орлова"]

TRANSLIT = dict(zip(
    "абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
    ["a", "b", "v", "g", "d", "e", "e", "zh", "z", "i", "y", "k", "l", "m", "n", "o", "p",
     "r", "s", "t", "u", "f", "kh", "ts", "ch", "sh", "shch", "", "y", "", "e", "yu", "ya"]))

_WRAPPED = re.compile(r"^\s*(\[|Ход\s*1)")


def typos(t, rng):
    words = t.split(" ")
    cand = [i for i, w in enumerate(words) if len(w) > 3 and w.isalpha()]
    for i in rng.sample(cand, min(len(cand), rng.randint(1, 3))):
        w = list(words[i])
        j = rng.randrange(len(w) - 1)
        op = rng.choice(("swap", "drop", "dup"))
        if op == "swap":
            w[j], w[j + 1] = w[j + 1], w[j]
        elif op == "drop":
            del w[j]
        else:
            w.insert(j, w[j])
        words[i] = "".join(w)
    return " ".join(words)


def case(t, rng):
    return t.lower() if rng.random() < 0.7 else t.upper()


def politeness(t, rng):
    if _WRAPPED.match(t):
        return t + rng.choice(SUFFIXES)
    p = rng.choice(PREFIXES)
    return p + (t[0].lower() + t[1:] if t else t) + (rng.choice(SUFFIXES) if rng.random() < 0.5 else "")


def punctuation(t, rng):
    if rng.random() < 0.5:
        return re.sub(r"[.,;:!?«»\"]", "", t)
    return re.sub(r"([.!?])", r"\1\1", t)


def entities(t, rng):
    t = re.sub(r"\d{2,}", lambda m: "".join(rng.choice("0123456789") for _ in m.group(0)), t)
    for s in SURNAMES:
        if s in t:
            t = t.replace(s, rng.choice(SURNAMES))
    return t


def translit(t, rng):
    out = []
    for c in t:
        low = c.lower()
        if low in TRANSLIT:
            r = TRANSLIT[low]
            out.append(r.capitalize() if c.isupper() else r)
        else:
            out.append(c)
    return "".join(out)


_CYR2LAT = str.maketrans("аеорсухАВЕКМНОРСТХ", "aeopcyxABEKMHOPCTX")


def homoglyphs(t, rng):
    words = t.split(" ")
    cand = [i for i, w in enumerate(words) if len(w) > 3 and any("\u0400" <= c <= "\u04ff" for c in w)]
    for i in rng.sample(cand, min(len(cand), rng.randint(1, 3))):
        w = list(words[i])
        pos = [j for j, c in enumerate(w) if c.translate(_CYR2LAT) != c]
        for j in rng.sample(pos, max(1, len(pos) // 2)) if pos else []:
            w[j] = w[j].translate(_CYR2LAT)
        words[i] = "".join(w)
    return " ".join(words)


TRANSFORMS = [(typos, 3), (case, 1), (politeness, 3), (punctuation, 2), (entities, 2), (translit, 1),
              (homoglyphs, 2)]


def augment(text, rng, n):
    funcs, weights = zip(*TRANSFORMS)
    variants = set()
    for _ in range(n * 4):
        if len(variants) >= n:
            break
        t = text
        for f in set(rng.choices(funcs, weights=weights, k=rng.randint(1, 2))):
            t = f(t, rng)
        t = t.strip()
        if t and t != text:
            variants.add(t)
    return list(variants)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--n", type=int, default=8, help="вариантов на строку")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    df = pd.read_csv(args.csv)
    if "group" not in df.columns:
        raise ValueError("Нужна колонка group (её создаёт convert_promptsc.py)")

    rows = []
    for _, r in df.iterrows():
        for v in augment(str(r["text"]), rng, args.n):
            rows.append({**r.to_dict(), "text": v, "augmented": True,
                         "source": f"{r.get('source', 'ru')}_aug"})
    aug = pd.DataFrame(rows)
    out = pd.concat([df.assign(augmented=False), aug], ignore_index=True)

    path = Path(args.csv).with_name(Path(args.csv).stem + "_aug.csv")
    out.to_csv(path, index=False)
    print(f"Исходных: {len(df)}, вариантов: {len(aug)}, всего: {len(out)} -> {path}")
    print(out.groupby(["augmented", "label"]).size().to_string())


if __name__ == "__main__":
    main()
