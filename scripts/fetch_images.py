from __future__ import annotations

import io
import urllib.request
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "input" / "raw"
PGM_DIR = ROOT / "data" / "input" / "pgm"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PGM_DIR.mkdir(parents=True, exist_ok=True)

SOURCES = {
    "lena": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
    "baboon": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/baboon.jpg",
    "fruits": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/fruits.jpg",
}

SIZES = [512, 1024, 2048]


def download(url: str) -> bytes:
    with urllib.request.urlopen(url) as resp:
        return resp.read()


def to_grayscale_pgm(img: Image.Image, path: Path) -> None:
    gray = img.convert("L")
    gray.save(path, format="PPM")


def main() -> None:
    for name, url in SOURCES.items():
        raw_path = RAW_DIR / f"{name}.jpg"
        if not raw_path.exists():
            raw_path.write_bytes(download(url))

        img = Image.open(raw_path)
        for size in SIZES:
            resized = img.resize((size, size), resample=Image.BICUBIC)
            out_path = PGM_DIR / f"{name}_{size}.pgm"
            to_grayscale_pgm(resized, out_path)

    print(f"Downloaded sources to {RAW_DIR}")
    print(f"Generated PGM files in {PGM_DIR}")


if __name__ == "__main__":
    main()
