from device import describe, get_device, setup_cpu

import argparse
import re
from pathlib import Path

import pandas as pd
import torch
from transformers import MarianMTModel, MarianTokenizer

from config import Config
from preprocessing import cyrillic_ratio

MODEL_ID = "Helsinki-NLP/opus-mt-en-ru"
_SENT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def split_sentences(text: str, max_chars: int = 400):
    parts, buf = [], ""
    for s in _SENT_RE.split(text):
        if not s.strip():
            continue
        if len(buf) + len(s) < max_chars:
            buf = f"{buf} {s}".strip()
        else:
            if buf:
                parts.append(buf)
            buf = s[:max_chars * 2]
    if buf:
        parts.append(buf)
    return parts or [text[:max_chars]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    en = df[df["text"].astype(str).map(cyrillic_ratio) < 0.3].reset_index(drop=True)
    if args.limit:
        en = en.sample(min(args.limit, len(en)), random_state=42).reset_index(drop=True)
    print(f"К переводу: {len(en)} строк")

    cfg = Config()
    device = get_device(cfg.device)
    if device == "cpu":
        setup_cpu(cfg.num_threads)
    print(f"[device] перевод на: {describe(device)}")
    tok = MarianTokenizer.from_pretrained(MODEL_ID)
    model = MarianMTModel.from_pretrained(MODEL_ID).to(device).eval()

    sents, owner = [], []
    for i, t in enumerate(en["text"].astype(str)):
        for s in split_sentences(t):
            sents.append(s)
            owner.append(i)

    translated = []
    with torch.no_grad():
        for b in range(0, len(sents), args.batch_size):
            batch = sents[b:b + args.batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            gen = model.generate(**enc, max_new_tokens=512, num_beams=2)
            translated += tok.batch_decode(gen, skip_special_tokens=True)
            if (b // args.batch_size) % 20 == 0:
                print(f"  {min(b + args.batch_size, len(sents))}/{len(sents)} предложений")

    joined = [""] * len(en)
    for i, t in zip(owner, translated):
        joined[i] = f"{joined[i]} {t}".strip()

    src = en["source"].astype(str) if "source" in en.columns else "public"
    out = en.assign(text=joined, translated=True, source=src + "_ru")
    out = out[out["text"].str.len() > 0]
    path = Path(args.csv).with_name(Path(args.csv).stem + "_ru.csv")
    out.to_csv(path, index=False)
    print(f"Сохранено {len(out)} строк -> {path}")
    print(out["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
