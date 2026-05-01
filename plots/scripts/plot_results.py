from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
data_path = ROOT / "benchmarks" / "raw" / "benchmarks.csv"
output_dir = ROOT / "plots" / "output"
output_dir.mkdir(parents=True, exist_ok=True)

if not data_path.exists():
	raise SystemExit(f"Benchmark data not found: {data_path}")

df = pd.read_csv(data_path)

sns.set_theme(style="whitegrid")


def add_bar_labels(ax, fmt="{:.3g}"):
	for container in ax.containers:
		ax.bar_label(container, fmt=fmt, padding=3, fontsize=8)

def plot_time(df_subset, title, filename, xcol):
	plt.figure(figsize=(8, 5))
	sns.lineplot(data=df_subset, x=xcol, y="seconds", hue="implementation", marker="o")
	plt.title(title)
	plt.tight_layout()
	plt.savefig(output_dir / filename, dpi=200)
	plt.close()


def plot_speedup(df_subset, title, filename, xcol):
	plt.figure(figsize=(8, 5))
	sns.lineplot(data=df_subset, x=xcol, y="speedup", hue="implementation", marker="o")
	plt.title(title)
	plt.tight_layout()
	plt.savefig(output_dir / filename, dpi=200)
	plt.close()


def plot_efficiency(df_subset, title, filename, xcol):
	plt.figure(figsize=(8, 5))
	sns.lineplot(data=df_subset, x=xcol, y="efficiency", hue="implementation", marker="o")
	plt.title(title)
	plt.tight_layout()
	plt.savefig(output_dir / filename, dpi=200)
	plt.close()


omp_df = df[df["implementation"] == "openmp"].copy()
mpi_df = df[df["implementation"] == "mpi"].copy()

if not omp_df.empty:
	plot_time(omp_df, "OpenMP Runtime vs Threads", "openmp_time.png", "threads")
	plot_speedup(omp_df, "OpenMP Speedup vs Threads", "openmp_speedup.png", "threads")
	plot_efficiency(omp_df, "OpenMP Efficiency vs Threads", "openmp_efficiency.png", "threads")

if not mpi_df.empty:
	plot_time(mpi_df, "MPI Runtime vs Ranks", "mpi_time.png", "ranks")
	plot_speedup(mpi_df, "MPI Speedup vs Ranks", "mpi_speedup.png", "ranks")
	plot_efficiency(mpi_df, "MPI Efficiency vs Ranks", "mpi_efficiency.png", "ranks")

if "image" in df.columns:
	best = (
		df.copy()
		.assign(config=lambda d: d[["threads", "ranks"]].astype(str).agg("/".join, axis=1))
		.groupby(["image", "implementation"], as_index=False)
		.agg(seconds=("seconds", "min"))
	)

	seq = best[best["implementation"] == "sequential"]["seconds"].rename("seq_seconds")
	seq_map = dict(zip(best[best["implementation"] == "sequential"]["image"], seq))
	best["speedup"] = best.apply(lambda r: seq_map.get(r["image"], 1.0) / r["seconds"], axis=1)

	plt.figure(figsize=(9, 5))
	ax = sns.barplot(data=best, x="image", y="seconds", hue="implementation")
	plt.title("Best Runtime by Architecture (per image)")
	add_bar_labels(ax, fmt="{:.3g}")
	plt.tight_layout()
	plt.savefig(output_dir / "arch_time_comparison.png", dpi=200)
	plt.close()

	plt.figure(figsize=(9, 5))
	ax = sns.barplot(data=best, x="image", y="speedup", hue="implementation")
	plt.title("Best Speedup by Architecture (per image)")
	add_bar_labels(ax, fmt="{:.2g}")
	plt.tight_layout()
	plt.savefig(output_dir / "arch_speedup_comparison.png", dpi=200)
	plt.close()

	cpu_only = best[best["implementation"].isin(["sequential", "openmp", "mpi"])]
	if not cpu_only.empty:
		plt.figure(figsize=(9, 5))
		ax = sns.barplot(data=cpu_only, x="image", y="seconds", hue="implementation")
		plt.title("CPU-only Runtime Comparison")
		add_bar_labels(ax, fmt="{:.3g}")
		plt.tight_layout()
		plt.savefig(output_dir / "cpu_only_runtime.png", dpi=200)
		plt.close()

		plt.figure(figsize=(9, 5))
		ax = sns.barplot(data=cpu_only, x="image", y="speedup", hue="implementation")
		plt.title("CPU-only Speedup Comparison")
		add_bar_labels(ax, fmt="{:.2g}")
		plt.tight_layout()
		plt.savefig(output_dir / "cpu_only_speedup.png", dpi=200)
		plt.close()

	gpu_only = best[best["implementation"].isin(["cuda", "openmp_target", "mpi_cuda"])]
	if not gpu_only.empty:
		plt.figure(figsize=(9, 5))
		ax = sns.barplot(data=gpu_only, x="image", y="seconds", hue="implementation")
		plt.title("GPU-only Runtime Comparison")
		add_bar_labels(ax, fmt="{:.3g}")
		plt.tight_layout()
		plt.savefig(output_dir / "gpu_only_runtime.png", dpi=200)
		plt.close()

		plt.figure(figsize=(9, 5))
		ax = sns.barplot(data=gpu_only, x="image", y="speedup", hue="implementation")
		plt.title("GPU-only Speedup Comparison")
		add_bar_labels(ax, fmt="{:.2g}")
		plt.tight_layout()
		plt.savefig(output_dir / "gpu_only_speedup.png", dpi=200)
		plt.close()

	cpu_best = (
		best[best["implementation"].isin(["sequential", "openmp", "mpi"])]
		.groupby("image", as_index=False)
		.agg(seconds=("seconds", "min"))
		.assign(implementation="cpu_best")
	)
	cuda_best = best[best["implementation"] == "cuda"].copy()

	if not cuda_best.empty:
		cpu_gpu = pd.concat([cpu_best, cuda_best], ignore_index=True)
		plt.figure(figsize=(9, 5))
		ax = sns.barplot(data=cpu_gpu, x="image", y="seconds", hue="implementation")
		plt.title("CPU Best vs CUDA Runtime")
		add_bar_labels(ax, fmt="{:.3g}")
		plt.tight_layout()
		plt.savefig(output_dir / "cpu_vs_cuda_runtime.png", dpi=200)
		plt.close()

		merged = cpu_best.merge(cuda_best, on="image", suffixes=("_cpu", "_cuda"))
		merged["cpu_over_cuda"] = merged["seconds_cpu"] / merged["seconds_cuda"]
		plt.figure(figsize=(9, 5))
		ax = sns.barplot(data=merged, x="image", y="cpu_over_cuda")
		plt.title("CPU Best / CUDA Speedup Ratio")
		add_bar_labels(ax, fmt="{:.2g}")
		plt.tight_layout()
		plt.savefig(output_dir / "cpu_vs_cuda_speedup.png", dpi=200)
		plt.close()

print(f"Plots written to {output_dir}")
