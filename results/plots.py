"""Plot experiment results from JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def plot_main_results(results: dict, output: str = "results/main_auc.png") -> None:
    models = list(results)
    compressions = sorted({c for m in results.values() for s in m.values() for c in s})
    x = np.arange(len(models))
    width = 0.8 / max(1, len(compressions))
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, c in enumerate(compressions):
        means, stds = [], []
        for model in models:
            vals = [float(s[c]) for s in results[model].values() if c in s]
            means.append(np.mean(vals) if vals else np.nan)
            stds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
        ax.bar(x + (i - (len(compressions)-1)/2)*width, means, width, yerr=stds, capsize=3, label=c)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("AUC")
    ax.set_title("FreqLoRA-CLIP compression robustness")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    plt.close(fig)
    print(f"Wrote {output}")


def plot_k_sweep(results: dict, output: str = "results/k_sweep.png") -> None:
    """Plot model AUC versus K when results are model -> K -> metric."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for model, k_results in results.items():
        ks = sorted(k_results, key=lambda v: int(v))
        vals = [k_results[k] if isinstance(k_results[k], (int, float)) else k_results[k].get("auc", np.nan) for k in ks]
        ax.plot([int(k) for k in ks], vals, marker="o", label=model)
    ax.set_xlabel("K shots per class")
    ax.set_ylabel("AUC")
    ax.set_title("Few-shot K sweep")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    plt.close(fig)
    print(f"Wrote {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/main_results.json")
    parser.add_argument("--output", default="results/main_auc.png")
    args = parser.parse_args()
    plot_main_results(load_json(args.input), args.output)
