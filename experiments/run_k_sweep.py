#!/usr/bin/env python3
"""K-shot sweep: evaluate M4 for K={1,5,10,20} over multiple seeds."""

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


def run(image_dir="data/processed/images", ks=(1, 5, 10, 20), seeds=(0,1,2,3,4),
        compression="C40", rank=8, epochs=100, batch_size=64):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}
    test_loader = get_dataloader(image_dir, split="test", batch_size=batch_size,
                                 compression_level=compression, seed=0)
    for k in ks:
        results[str(k)] = {}
        for seed in seeds:
            set_seed(seed)
            support_images, support_labels = get_few_shot_data(image_dir, k=k, seed=seed, device=device)
            model = get_model("M4", rank=rank, device=device)
            train_model(model, support_images, support_labels, device=device, epochs=epochs)
            metrics = Evaluator(model, device=device).evaluate(test_loader)
            results[str(k)][f"seed_{seed}"] = metrics
            print(f"K={k} seed={seed} {compression} AUC={metrics['auc']:.4f}")
    save_json(results, "results/k_sweep.json")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image-dir", default="data/processed/images")
    p.add_argument("--compression", default="C40", choices=["C0", "C23", "C40"])
    p.add_argument("--epochs", type=int, default=100)
    args = p.parse_args()
    run(args.image_dir, compression=args.compression, epochs=args.epochs)
