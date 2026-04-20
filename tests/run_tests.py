from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "results" / "images"


def load_pgm(path: Path):
    with path.open("rb") as f:
        magic = f.readline().strip()
        if magic != b"P5":
            raise ValueError(f"Unsupported PGM format: {magic}")

        def next_token() -> bytes:
            token = f.readline()
            while token.startswith(b"#"):
                token = f.readline()
            return token

        dims = next_token().split()
        while len(dims) < 2:
            dims += next_token().split()
        width, height = map(int, dims)
        maxval = int(next_token())
        if maxval > 255:
            raise ValueError("Only 8-bit PGM supported")

        data = f.read(width * height)
        if len(data) != width * height:
            raise ValueError("Unexpected PGM length")

    return width, height, data


def compare(a: Path, b: Path) -> float:
    wa, ha, da = load_pgm(a)
    wb, hb, db = load_pgm(b)
    if wa != wb or ha != hb:
        raise ValueError(f"Shape mismatch: {a} vs {b}")
    max_diff = 0
    for va, vb in zip(da, db):
        max_diff = max(max_diff, abs(va - vb))
    return max_diff


def main() -> int:
    pairs = [
        (IMAGES / "seq_blur.pgm", IMAGES / "omp_blur.pgm"),
        (IMAGES / "seq_blur.pgm", IMAGES / "mpi_blur.pgm"),
        (IMAGES / "seq_blur.pgm", IMAGES / "cuda_blur.pgm"),
        (IMAGES / "seq_blur.pgm", IMAGES / "omp_target_blur.pgm"),
        (IMAGES / "seq_blur.pgm", IMAGES / "mpi_cuda_blur.pgm"),
        (IMAGES / "seq_sobel.pgm", IMAGES / "omp_sobel.pgm"),
        (IMAGES / "seq_sobel.pgm", IMAGES / "mpi_sobel.pgm"),
        (IMAGES / "seq_sobel.pgm", IMAGES / "cuda_sobel.pgm"),
        (IMAGES / "seq_sobel.pgm", IMAGES / "omp_target_sobel.pgm"),
        (IMAGES / "seq_sobel.pgm", IMAGES / "mpi_cuda_sobel.pgm"),
    ]

    failures = 0
    for a, b in pairs:
        if not a.exists() or not b.exists():
            continue
        diff = compare(a, b)
        print(f"Compare {a.name} vs {b.name}: max diff = {diff}")
        if diff > 1:
            failures += 1

    if failures:
        print(f"FAIL: {failures} comparisons exceeded tolerance")
        return 1

    print("PASS: outputs are consistent within tolerance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
