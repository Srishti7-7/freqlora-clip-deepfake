"""Metric aggregation helpers for multi-seed experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np


def aggregate_seeds(values: list[float] | np.ndarray, ddof: int = 1) -> dict[str, float]:
    """Return mean, standard deviation, and number of seeds."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    std = float(arr.std(ddof=ddof)) if arr.size > ddof else 0.0
    return {"mean": float(arr.mean()), "std": std, "n": int(arr.size)}


def aggregate_result_dict(results: dict[str, Any], metric_key: str = "auc") -> dict[str, Any]:
    """Aggregate seed-level results stored as model -> seed -> compression -> metric."""
    output: dict[str, Any] = {}
    for model, seed_results in results.items():
        compressions = sorted({c for r in seed_results.values() for c in r})
        output[model] = {}
        for compression in compressions:
            vals = [r[compression] for r in seed_results.values() if metric_key in r or compression in r]
            # run_main stores AUC directly under each compression.
            output[model][compression] = aggregate_seeds(vals)
    return output


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
