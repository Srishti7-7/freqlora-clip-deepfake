#!/usr/bin/env python3
"""Cross-dataset evaluation on a Celeb-DF v2 frame directory.

Expected layout for the evaluation directory: real/*.png and fake/*.png.
The model is trained on FF++ support frames and evaluated without adaptation.
"""

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


def run(train_dir="data/processed/images", cross_dataset_dir="data/processed/celeb_df_v2",
        k=5, seed=0, rank=8, epochs=100):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(seed)
    x, y = get_few_shot_data(train_dir, k=k, seed=seed, device=device)
    model = get_model("M4", rank=rank, device=device)
    train_model(model, x, y, device=device, epochs=epochs)
    loader = get_dataloader(cross_dataset_dir, split="all", compression_level="C0", seed=0)
    metrics = Evaluator(model, device=device).evaluate(loader)
    save_json(metrics, "results/cross_dataset.json")
    print("Celeb-DF v2:", metrics)
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train-dir", default="data/processed/images")
    p.add_argument("--cross-dataset-dir", default="data/processed/celeb_df_v2")
    p.add_argument("--epochs", type=int, default=100)
    args = p.parse_args()
    run(args.train_dir, args.cross_dataset_dir, epochs=args.epochs)
