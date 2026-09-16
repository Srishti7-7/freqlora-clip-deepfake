# FreqLoRA-CLIP: Compression-Robust Deepfake Detection

Research codebase for few-shot deepfake detection using CLIP visual features,
low-rank residual adaptation, and DCT-based frequency features.

## Models

- **M1** — CLIP-only baseline
- **M2** — CLIP + low-rank residual adapter
- **M3** — CLIP + DCT frequency features
- **M4** — CLIP + low-rank adapter + DCT frequency features

> The current M2/M4 implementation applies a low-rank residual adapter to the
> final CLIP visual embedding; it is not a transformer-layer LoRA implementation.

## Dataset

Expected extracted-frame layout:

```text
data/processed/images/
├── real/
└── fake/
```

The dataset is intentionally excluded by `.gitignore`. **Do not push the
dataset or model checkpoints to GitHub.**

## Installation

```bash
pip install -r requirements.txt
```

or:

```bash
python setup.py
```

## Main experiment

After the dataset is available:

```bash
python experiments/run_main.py
```

## Important status notes

The supplied source includes working implementations for the four model
architectures, data loader, trainer, evaluator, main experiment runner, and
setup script.

Several files listed in the intended structure did not have implementations
in the supplied source; they are included as explicit placeholders rather than
being represented as completed code.

Also, the supplied `run_main.py` labels C23/C40 evaluations but currently calls
the same test loader without passing a compression condition. Therefore,
distinct C23/C40 evaluation should not be treated as implemented until the
compression pipeline is added.

No experimental results are included. Results should only be committed after
experiments are actually run.

## License

MIT
