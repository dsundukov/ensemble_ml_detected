import argparse
import hashlib
import json
import sys
from pathlib import Path

CHUNK = 8 * 1024 * 1024


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=".", help="куда положить собранные архивы")
    args = ap.parse_args()
    here = Path(__file__).resolve().parent
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((here / "manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        name = item["name"]
        if not item["parts"]:
            src = here / name
            if not src.exists() or sha256(src) != item["sha256"]:
                sys.exit(f"{name}: нет файла или он повреждён — скачайте заново")
            print(f"{name}: OK (целиком)")
            continue
        for p in item["parts"]:
            path = here / p["name"]
            if not path.exists():
                sys.exit(f"Нет части {p['name']} — скачайте все части из manifest.json")
            if path.stat().st_size < 1024 and path.read_bytes().startswith(b"version https://git-lfs"):
                sys.exit(f"{p['name']} — указатель Git LFS, выполните: git lfs pull")
            if path.stat().st_size != p["size"] or sha256(path) != p["sha256"]:
                sys.exit(f"Часть {p['name']} повреждена или скачана не полностью — скачайте её заново")
        out = dest / name
        with open(out, "wb") as w:
            for p in item["parts"]:
                with open(here / p["name"], "rb") as f:
                    for b in iter(lambda: f.read(CHUNK), b""):
                        w.write(b)
        if sha256(out) != item["sha256"]:
            out.unlink()
            sys.exit(f"{name}: контрольная сумма не совпала — скачайте части заново")
        print(f"{name}: собран из {len(item['parts'])} частей, {item['size'] / 1e6:.0f} МБ, SHA-256 OK -> {out}")
    print("Готово. Распакуйте архивы обычным unzip.")


if __name__ == "__main__":
    main()
