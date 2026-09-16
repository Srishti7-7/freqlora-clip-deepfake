"""Generate LaTeX tables from saved experiment JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main_table_latex(results: dict, digits: int = 4) -> str:
    """Create a model x compression AUC table from run_main.py output."""
    compressions = sorted({c for model in results.values() for seed in model.values() for c in seed})
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Few-shot deepfake detection AUC across compression levels.}",
        r"\begin{tabular}{l" + "c" * len(compressions) + "}",
        r"\toprule",
        "Model & " + " & ".join(compressions) + r" \\",
        r"\midrule",
    ]
    for model, seed_results in results.items():
        cells = []
        for c in compressions:
            vals = [float(seed[c]) for seed in seed_results.values() if c in seed]
            if vals:
                mean = sum(vals) / len(vals)
                if len(vals) > 1:
                    import statistics
                    std = statistics.stdev(vals)
                else:
                    std = 0.0
                cells.append(f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}")
            else:
                cells.append("--")
        lines.append(model + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def write_table(input_json: str, output_tex: str) -> None:
    text = main_table_latex(_load(input_json))
    Path(output_tex).parent.mkdir(parents=True, exist_ok=True)
    Path(output_tex).write_text(text, encoding="utf-8")
    print(f"Wrote {output_tex}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/main_results.json")
    parser.add_argument("--output", default="results/main_table.tex")
    args = parser.parse_args()
    write_table(args.input, args.output)
