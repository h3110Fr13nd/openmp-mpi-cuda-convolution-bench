from __future__ import annotations

import csv
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "build"
OUT = ROOT / "benchmarks" / "raw"
PGM_DIR = ROOT / "data" / "input" / "pgm"
OUT.mkdir(parents=True, exist_ok=True)


def run_command(cmd: List[str], env: Dict[str, str] | None = None) -> float:
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{proc.stderr}")
    for line in proc.stdout.splitlines():
        if line.startswith("sequential_elapsed_seconds"):
            return float(line.split(",")[1])
        if line.startswith("openmp_elapsed_seconds"):
            return float(line.split(",")[1])
        if line.startswith("openmp_target_elapsed_seconds"):
            return float(line.split(",")[1])
        if line.startswith("mpi_elapsed_seconds"):
            return float(line.split(",")[1])
        if line.startswith("mpi_cuda_elapsed_seconds"):
            return float(line.split(",")[1])
        if line.startswith("cuda_elapsed_seconds"):
            return float(line.split(",")[1])
    raise RuntimeError(f"No timing info found in output: {proc.stdout}")


def parse_omp_target_devices(output: str) -> int | None:
    for line in output.splitlines():
        if line.startswith("openmp_target_devices"):
            return int(line.split(",")[1])
    return None


def probe_omp_target_device(cmd: List[str], env: Dict[str, str]) -> bool:
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        print("OpenMP target probe failed; skipping openmp_target benchmarks.")
        print(proc.stderr.strip())
        return False
    devices = parse_omp_target_devices(proc.stdout)
    if devices is None or devices <= 0:
        print("OpenMP target offload not detected; skipping openmp_target benchmarks.")
        return False
    return True


def median_time(cmd: List[str], env: Dict[str, str] | None = None, runs: int = 3) -> float:
    times = [run_command(cmd, env=env) for _ in range(runs)]
    return statistics.median(times)


def exe(name: str) -> Path:
    return BUILD / name


def read_pgm_shape(path: Path) -> Tuple[int, int]:
    with path.open("rb") as f:
        magic = f.readline().strip()
        if magic != b"P5":
            raise RuntimeError(f"Unsupported PGM format: {path}")

        def next_token() -> bytes:
            token = f.readline()
            while token.startswith(b"#"):
                token = f.readline()
            return token

        dims = next_token().split()
        while len(dims) < 2:
            dims += next_token().split()
        width, height = map(int, dims)
        _ = next_token()
    return width, height


def main() -> None:
    kernel_by_filter = {
        "blur": 5,
        "sobel": 3,
    }
    inputs = [
        PGM_DIR / "lena_512.pgm",
        PGM_DIR / "lena_1024.pgm",
        PGM_DIR / "lena_2048.pgm",
        PGM_DIR / "baboon_512.pgm",
        PGM_DIR / "fruits_512.pgm",
    ]

    records: List[dict] = []

    if not exe("seq_convolution").exists():
        print("Sequential executable not found.")
        return

    for filter_name, kernel in kernel_by_filter.items():
        omp_target_available = False
        if exe("omp_target_convolution").exists():
            probe_input = next((p for p in inputs if p.exists()), None)
            if probe_input is None:
                print("No input images available to probe OpenMP target offload.")
            else:
                env = os.environ.copy()
                env["OMP_TARGET_OFFLOAD"] = "MANDATORY"
                env.setdefault("OMP_DEFAULT_DEVICE", "0")
                probe_cmd = [
                    str(exe("omp_target_convolution")),
                    "--filter",
                    filter_name,
                    "--kernel-size",
                    str(kernel),
                    "--input",
                    str(probe_input),
                    "--no-output",
                    "--iterations",
                    "1",
                ]
                omp_target_available = probe_omp_target_device(probe_cmd, env)

        for input_path in inputs:
            if not input_path.exists():
                print(f"Missing input image: {input_path}")
                continue

            width, height = read_pgm_shape(input_path)
            image_name = input_path.stem

            cmd = [str(exe("seq_convolution")), "--filter", filter_name, "--kernel-size", str(kernel),
                   "--input", str(input_path), "--no-output"]
            seq_time = median_time(cmd)
            records.append({
                "implementation": "sequential",
                "threads": 1,
                "ranks": 1,
                "image": image_name,
                "width": width,
                "height": height,
                "filter": filter_name,
                "seconds": seq_time,
                "speedup": 1.0,
                "efficiency": 1.0,
            })

            if exe("omp_convolution").exists():
                for threads in [1, 2, 4, 8, 16]:
                    env = os.environ.copy()
                    env["OMP_NUM_THREADS"] = str(threads)
                    cmd = [str(exe("omp_convolution")), "--filter", filter_name, "--kernel-size", str(kernel),
                           "--input", str(input_path), "--no-output"]
                    omp_time = median_time(cmd, env=env)
                    speedup = seq_time / omp_time
                    records.append({
                        "implementation": "openmp",
                        "threads": threads,
                        "ranks": 1,
                        "image": image_name,
                        "width": width,
                        "height": height,
                        "filter": filter_name,
                        "seconds": omp_time,
                        "speedup": speedup,
                        "efficiency": speedup / threads,
                    })

            if exe("omp_target_convolution").exists() and omp_target_available:
                for threads in [1, 2, 4, 8, 16]:
                    env = os.environ.copy()
                    env["OMP_NUM_THREADS"] = str(threads)
                    env["OMP_TARGET_OFFLOAD"] = "MANDATORY"
                    env.setdefault("OMP_DEFAULT_DEVICE", "0")
                    cmd = [str(exe("omp_target_convolution")), "--filter", filter_name, "--kernel-size", str(kernel),
                           "--input", str(input_path), "--no-output"]
                    omp_t_time = median_time(cmd, env=env)
                    speedup = seq_time / omp_t_time
                    records.append({
                        "implementation": "openmp_target",
                        "threads": threads,
                        "ranks": 1,
                        "image": image_name,
                        "width": width,
                        "height": height,
                        "filter": filter_name,
                        "seconds": omp_t_time,
                        "speedup": speedup,
                        "efficiency": speedup / threads,
                    })

            if exe("mpi_convolution").exists():
                for ranks in [1, 2, 4, 8]:
                    cmd = ["mpirun", "-np", str(ranks), str(exe("mpi_convolution")),
                           "--filter", filter_name, "--kernel-size", str(kernel),
                           "--input", str(input_path), "--no-output"]
                    mpi_time = median_time(cmd)
                    speedup = seq_time / mpi_time
                    records.append({
                        "implementation": "mpi",
                        "threads": 1,
                        "ranks": ranks,
                        "image": image_name,
                        "width": width,
                        "height": height,
                        "filter": filter_name,
                        "seconds": mpi_time,
                        "speedup": speedup,
                        "efficiency": speedup / ranks,
                    })

            if exe("mpi_cuda_convolution").exists():
                for ranks in [1, 2, 4, 8]:
                    cmd = ["mpirun", "-np", str(ranks), str(exe("mpi_cuda_convolution")),
                           "--filter", filter_name, "--kernel-size", str(kernel),
                           "--input", str(input_path), "--no-output"]
                    mpi_cuda_time = median_time(cmd)
                    speedup = seq_time / mpi_cuda_time
                    records.append({
                        "implementation": "mpi_cuda",
                        "threads": 1,
                        "ranks": ranks,
                        "image": image_name,
                        "width": width,
                        "height": height,
                        "filter": filter_name,
                        "seconds": mpi_cuda_time,
                        "speedup": speedup,
                        "efficiency": speedup / ranks,
                    })

            if exe("cuda_convolution").exists():
                cmd = [str(exe("cuda_convolution")), "--filter", filter_name, "--kernel-size", str(kernel),
                       "--input", str(input_path), "--no-output"]
                cuda_time = median_time(cmd)
                speedup = seq_time / cuda_time
                records.append({
                    "implementation": "cuda",
                    "threads": "na",
                    "ranks": "na",
                    "image": image_name,
                    "width": width,
                    "height": height,
                    "filter": filter_name,
                    "seconds": cuda_time,
                    "speedup": speedup,
                    "efficiency": "na",
                })

    if not records:
        print("No benchmark records collected.")
        return

    out_path = OUT / "benchmarks.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
