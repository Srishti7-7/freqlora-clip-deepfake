# FreqLoRA-CLIP: Compression-Robust Few-Shot Deepfake Detection

Research codebase for a few-shot deepfake detection pipeline combining frozen CLIP visual embeddings, a low-rank residual adapter, and DCT-based frequency features for evaluation under H.264 compression.

## Models

- **M1** — CLIP baseline + linear classifier
- **M2** — CLIP + low-rank residual adapter
- **M3** — CLIP + DCT frequency features
- **M4** — CLIP + low-rank residual adapter + DCT frequency features

> Note: the supplied architecture implements the low-rank adapter on the final CLIP embedding. It is a low-rank residual adapter rather than transformer-layer LoRA. The repository keeps the original project terminology while documenting this distinction.

## Repository structure

```text
freqlora-clip-deepfake/
├── .gitignore
├── README.md
├── requirements.txt
├── setup.py
├── models/
│   ├── __init__.py
│   └── architectures.py
├── data/
│   ├── __init__.py
│   ├── loader.py
│   └── compression.py
├── training/
│   ├── __init__.py
│   ├── trainer.py
│   └── losses.py
├── evaluation/
│   ├── __init__.py
│   ├── evaluator.py
│   └── metrics.py
├── experiments/
│   ├── __init__.py
│   ├── run_main.py
│   ├── run_k_sweep.py
│   ├── run_ablations.py
│   ├── run_cross_dataset.py
│   └── utils.py
├── results/
│   ├── __init__.py
│   ├── tables.py
│   └── plots.py
└── checkpoints/
```

## Dataset

The loaders expect extracted frames in:

```text
data/processed/images/
├── real/*.png
└── fake/*.png
```

For compressed evaluation, generate:

```text
data/processed/images/
├── real/*.png
├── fake/*.png
├── C23/real/*.png
├── C23/fake/*.png
├── C40/real/*.png
└── C40/fake/*.png
```

`C0` means the original frames; `C23` and `C40` correspond to H.264 CRF 23 and CRF 40. FFmpeg is required to generate compressed frames.

Do **not** commit FaceForensics++ data, videos, or checkpoints to GitHub.

## Setup

```bash
pip install -r requirements.txt
python setup.py
```

The supplied `setup.py` also creates the expected data/results directories and downloads the CLIP checkpoint through the OpenAI CLIP package when run.

## Generate compressed evaluation data

```python
from data.compression import build_compressed_image_dataset

build_compressed_image_dataset(
    "data/processed/images",
    "data/processed/images"
)
```

For video-faithful experiments, compress each source video at CRF 23/40 and extract its frames before constructing the class directories.

## Main experiment

```bash
python experiments/run_main.py
```

The main experiment uses K=5 support examples per class, five seeds, and evaluates M1-M4 on C23 and C40. The experiment now passes the selected compression level to the dataloader rather than treating C23/C40 as labels only.

For a quick smoke run:

```bash
python experiments/run_main.py
```

and reduce `epochs` in code or call `run_main_experiment(epochs=5)` when testing locally.

## Additional experiments

```bash
python experiments/run_k_sweep.py
python experiments/run_ablations.py
python experiments/run_cross_dataset.py
```

The cross-dataset script expects a frame directory with `real/*.png` and `fake/*.png`, for example `data/processed/celeb_df_v2/`.

## Results

Generate a LaTeX table from the main experiment:

```bash
python results/tables.py
```

Generate the main AUC plot:

```bash
python results/plots.py --input results/main_results.json --output results/main_auc.png
```

No experimental performance values are hard-coded in this repository. Run the experiments to produce them.

## License

MIT License.
