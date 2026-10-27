import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

_SIDE_FILES = (
    "config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "vocab.txt", "sentencepiece.bpe.model", "added_tokens.json",
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--out", default="ensemble")
    args = ap.parse_args()

    src, out = Path(args.artifacts), Path(args.out)
    cfg = json.loads((src / "ensemble_config.json").read_text(encoding="utf-8"))
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    tensors = {}

    tdir = src / "models" / "tfidf"
    tcfg = json.loads((tdir / "config.json").read_text(encoding="utf-8"))
    for k, v in load_file(str(tdir / "model.safetensors")).items():
        tensors[f"tfidf.{k}"] = v
    for v in tcfg["vectorizers"]:
        shutil.copy(tdir / f"vocab_{v['name']}.json", out / f"vocab_{v['name']}.json")

    for t in cfg["transformers"]:
        mdir = src / "models" / t["name"]
        weights = mdir / "model.safetensors"
        if not weights.exists():
            raise FileNotFoundError(f"Нет {weights}: обучение {t['name']} не завершилось")
        for k, v in load_file(str(weights)).items():
            tensors[f"{t['name']}.{k}"] = np.ascontiguousarray(v)
        (out / t["name"]).mkdir()
        for f in _SIDE_FILES:
            if (mdir / f).exists():
                shutil.copy(mdir / f, out / t["name"] / f)

    for k, v in load_file(str(src / "models" / "meta" / "model.safetensors")).items():
        tensors[f"meta.{k}"] = v

    save_file(tensors, str(out / "model.safetensors"),
              metadata={"format": "prompt_attack_ensemble", "version": "1"})

    ens_cfg = {
        **cfg,
        "tfidf_vectorizers": tcfg["vectorizers"],
        "components": ["tfidf"] + [t["name"] for t in cfg["transformers"]] + ["meta"],
    }
    (out / "ensemble_config.json").write_text(
        json.dumps(ens_cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    size_mb = (out / "model.safetensors").stat().st_size / 1e6
    print(f"Готово: {out}/model.safetensors ({size_mb:.0f} МБ, тензоров: {len(tensors)})")
    for c in ens_cfg["components"]:
        n = sum(1 for k in tensors if k.startswith(c + "."))
        print(f"  {c:<14} {n} тензоров")


if __name__ == "__main__":
    main()
