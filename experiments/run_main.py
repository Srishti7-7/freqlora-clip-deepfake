#!/usr/bin/env python3
"""
experiments/run_main.py - Main experiment (Table 1)
Trains M1, M2, M3, M4 on C23, evaluates on C23 and C40
K=5, 5 seeds
"""

import sys
sys.path.insert(0, '/'.join(__file__.split('/')[:-2]))

import torch
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

from models.architectures import get_model, count_trainable_params
from data.loader import get_few_shot_data, get_dataloader
from training.trainer import train_model
from evaluation.evaluator import Evaluator
from experiments.utils import set_seed, save_json

def run_main_experiment(epochs=100):
    """Main experiment: 4 models × 2 compressions × 5 seeds"""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    
    models_to_test = ["M1", "M2", "M3", "M4"]
    compressions = ["C23", "C40"]
    seeds = [0, 1, 2, 3, 4]
    k_value = 5
    
    # Results storage
    results = {}
    
    # Assume FF++ frames extracted to data/processed/images/
    image_dir = Path("data/processed/images")
    
    print("="*70)
    print("MAIN EXPERIMENT: All Models × All Compressions × All Seeds")
    print("="*70)
    
    for model_name in models_to_test:
        print(f"\n{'='*70}")
        print(f"Model: {model_name}")
        print(f"{'='*70}\n")
        
        model_results = {}
        
        for seed in seeds:
            print(f"Seed {seed}:")
            set_seed(seed)
            
            seed_results = {}
            
            # Get few-shot data (always from original uncompressed)
            support_images, support_labels = get_few_shot_data(
                image_dir, k=k_value, seed=seed, device=device
            )
            
            # Train model
            model = get_model(model_name, device=device)
            print(f"  Trainable params: {count_trainable_params(model)}")
            
            train_model(model, support_images, support_labels, device=device, epochs=epochs)
            
            # Evaluate on each compression level
            for compression in compressions:
                print(f"    Evaluating on {compression}...", end=" ")
                
                # Get test loader
                test_loader = get_dataloader(image_dir, split='test', compression_level=compression, seed=seed)
                
                # Evaluate
                evaluator = Evaluator(model, device=device)
                metrics = evaluator.evaluate(test_loader)
                
                seed_results[compression] = metrics['auc']
                print(f"AUC: {metrics['auc']:.4f}")
            
            model_results[f"seed_{seed}"] = seed_results
        
        results[model_name] = model_results
    
    # Save results
    Path("results").mkdir(exist_ok=True)
    save_json(results, "results/main_results.json")
    
    print("\n" + "="*70)
    print("Results saved to results/main_results.json")
    print("="*70)
    
    return results


if __name__ == "__main__":
    run_main_experiment()
