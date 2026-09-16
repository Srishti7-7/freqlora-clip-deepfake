#!/usr/bin/env python3
"""Rank ablation for the low-rank adapter used by M4."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from models.architectures import get_model
from data.loader import get_few_shot_data, get_dataloader
from training.trainer import train_model
from evaluation.evaluator import Evaluator
from experiments.utils import set_seed, save_json


def run(image_dir="data/processed/images", ranks=(1,4,8,16,32), seeds=(0,1,2,3,4),
        compression="C40", k=5, epochs=100):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_loader = get_dataloader(image_dir, split="test", compression_level=compression, seed=0)
    results = {}
    for rank in ranks:
        results[str(rank)] = {}
        for seed in seeds:
            set_seed(seed)
            x, y = get_few_shot_data(image_dir, k=k, seed=seed, device=device)
            model = get_model("M4", rank=rank, device=device)
            train_model(model, x, y, device=device, epochs=epochs)
            metrics = Evaluator(model, device=device).evaluate(test_loader)
            results[str(rank)][f"seed_{seed}"] = metrics
            print(f"rank={rank} seed={seed} {compression} AUC={metrics['auc']:.4f}")
    save_json(results, "results/rank_ablation.json")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image-dir", default="data/processed/images")
    p.add_argument("--compression", default="C40", choices=["C0", "C23", "C40"])
    p.add_argument("--epochs", type=int, default=100)
    args = p.parse_args()
    run(args.image_dir, compression=args.compression, epochs=args.epochs)
