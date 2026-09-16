\# FreqLoRA-CLIP Training Guide



Operational guide for training and evaluating the FreqLoRA-CLIP experiment pipeline.



> \*\*Important:\*\* This repository contains the experiment implementation. Actual experimental results must come from real runs. Do not fabricate or fill missing AUC, accuracy, or other metric values.



\---



\## 1. Clone the Repository



```bash

git clone https://github.com/Srishti7-7/freqlora-clip-deepfake.git

cd freqlora-clip-deepfake

```



If the repository is already cloned:



```bash

git pull origin main

```



\---



\## 2. Create the Python Environment



Recommended Python version: \*\*3.10 or 3.11\*\*



\### Windows



```powershell

python -m venv .venv

.venv\\Scripts\\activate

```



\### Linux



```bash

python3 -m venv .venv

source .venv/bin/activate

```



Install dependencies:



```bash

pip install --upgrade pip

pip install -r requirements.txt

```



\---



\## 3. Check PyTorch and GPU



Run:



```bash

python -c "import torch; print('PyTorch:', torch.\_\_version\_\_); print('CUDA available:', torch.cuda.is\_available()); print('GPU:', torch.cuda.get\_device\_name(0) if torch.cuda.is\_available() else 'CPU')"

```



A CUDA-capable NVIDIA GPU is strongly preferred for practical training.



If CUDA is unavailable, the code may fall back to CPU, but training can be substantially slower.



\---



\# 4. Dataset



The project expects preprocessed image data and does \*\*not\*\* require datasets to be committed to GitHub.



The expected processed-data layout is:



```text

data/

└── processed/

&#x20;   └── images/

&#x20;       ├── real/

&#x20;       ├── fake/

&#x20;       ├── C23/

&#x20;       │   ├── real/

&#x20;       │   └── fake/

&#x20;       └── C40/

&#x20;           ├── real/

&#x20;           └── fake/

```



Do \*\*not\*\* upload FaceForensics++ or Celeb-DF datasets to GitHub.



The exact source-video download and preprocessing procedure is not fully specified by the repository code. Use the permitted dataset-access and preprocessing procedure used by the project team.



\---



\# 5. Compression Conditions



The experiment pipeline supports the following compression conditions:



\* \*\*C0\*\* — uncompressed/original condition

\* \*\*C23\*\* — compression condition

\* \*\*C40\*\* — stronger compression condition



The updated experiment pipeline passes the compression condition to the data loader.



Therefore, `C23` and `C40` should correspond to actual compressed data/selection rather than only being printed as labels.



Before full training, verify that the required C23 and C40 data are available in the expected locations.



\---



\# 6. Important Project Files



\### Model architectures



```text

models/architectures.py

```



Contains the model configurations used by the project, including M1–M4.



\### Data loading



```text

data/loader.py

```



\### Compression utilities



```text

data/compression.py

```



\### Training



```text

training/trainer.py

training/losses.py

```



\### Evaluation



```text

evaluation/evaluator.py

evaluation/metrics.py

```



\### Main experiment



```text

experiments/run\_main.py

```



\### Few-shot K sweep



```text

experiments/run\_k\_sweep.py

```



\### Ablation experiments



```text

experiments/run\_ablations.py

```



\### Cross-dataset experiment



```text

experiments/run\_cross\_dataset.py

```



\### Experiment utilities



```text

experiments/utils.py

```



\### Results



```text

results/tables.py

results/plots.py

```



\---



\# 7. Run a Sanity Check First



Do \*\*not\*\* immediately start the complete experiment.



First verify:



1\. Python environment works.

2\. Required packages import correctly.

3\. PyTorch works.

4\. GPU is detected if available.

5\. Dataset folders are visible.

6\. Images can be loaded.

7\. The model can be instantiated.

8\. A forward pass works.

9\. A short training/evaluation run completes without errors.



This prevents wasting GPU time because of a path, dependency, or dataset-format problem.



\---



\# 8. Main Experiment



The primary experiment script is:



```text

experiments/run\_main.py

```



Run:



```bash

python experiments/run\_main.py

```



Before starting a long run, check the configuration in `run\_main.py`.



Confirm:



\* Dataset path

\* Few-shot setting

\* Seeds

\* Compression conditions

\* Model configuration

\* Device

\* Output paths

\* Checkpoint paths



Do not change experimental settings without recording the change.



\---



\# 9. Model Configurations



The project evaluates the model configurations:



```text

M1

M2

M3

M4

```



The implementations are located in:



```text

models/architectures.py

```



The exact architecture definitions in the repository should be treated as the source of truth for the experiments.



\---



\# 10. Few-Shot Setting



The project evaluates few-shot learning.



The K values used by the K-sweep are:



```text

K = 1

K = 5

K = 10

K = 20

```



The few-shot sampler was updated so that the random seed affects sample selection.



Record the seed used for every run.



\---



\# 11. K-Sweep Experiment



Run:



```bash

python experiments/run\_k\_sweep.py

```



The intended sweep evaluates:



```text

K = 1

K = 5

K = 10

K = 20

```



using multiple seeds.



The purpose is to study how performance changes as the number of labeled examples increases.



Preserve the generated results and avoid overwriting previous experiments.



\---



\# 12. Ablation Experiment



Run:



```bash

python experiments/run\_ablations.py

```



The ablation pipeline evaluates configuration choices implemented by the repository, including the adapter/rank sweep.



Before running:



1\. Open the script.

2\. Check the configurations being evaluated.

3\. Record the exact values.

4\. Record the seeds.

5\. Keep the resulting raw outputs.



Do not modify the experiment protocol without documenting the modification.



\---



\# 13. Cross-Dataset Experiment



Run:



```bash

python experiments/run\_cross\_dataset.py

```



The experiment is intended to:



```text

Train → FaceForensics++

Evaluate → Celeb-DF v2

```



Make sure the evaluation dataset is available in the format expected by the loader.



Do not mix evaluation-dataset images into the training set.



\---



\# 14. Evaluation Metrics



Evaluation code:



```text

evaluation/evaluator.py

evaluation/metrics.py

```



Preserve the result from every individual seed.



When multiple seeds are actually run, results can be summarized as:



```text

Mean ± Standard Deviation

```



Do \*\*not\*\* report a standard deviation if only one run was performed.



Do \*\*not\*\* replace missing experimental results with values from another paper.



\---



\# 15. Generate Tables



Table-generation utilities are located at:



```text

results/tables.py

```



Run the table-generation utility after the required experiments have completed and the result files contain actual measurements.



The table utility only formats results. It does not generate experimental evidence.



\---



\# 16. Generate Plots



Plotting utilities are located at:



```text

results/plots.py

```



Use them after the experiments have generated actual result files.



The plotting utilities support the main experiment results and K-sweep visualization implemented in the repository.



\---



\# 17. Recommended Complete Workflow



Follow this order:



```text

Clone / Pull Repository

&#x20;         ↓

Create Virtual Environment

&#x20;         ↓

Install Requirements

&#x20;         ↓

Check GPU

&#x20;         ↓

Prepare Dataset

&#x20;         ↓

Verify Dataset Paths

&#x20;         ↓

Verify C23 / C40

&#x20;         ↓

Run Sanity Check

&#x20;         ↓

Run Main M1–M4 Experiments

&#x20;         ↓

Run K Sweep

&#x20;         ↓

Run Ablation Experiments

&#x20;         ↓

Run Cross-Dataset Experiment

&#x20;         ↓

Generate Tables

&#x20;         ↓

Generate Plots

&#x20;         ↓

Review Logs and Results

&#x20;         ↓

Back Up Checkpoints and Results

```



\---



\# 18. Reproducibility Record



For every experiment, record:



```text

Date:

Git commit:

Python version:

PyTorch version:

GPU:

Dataset version/source:

Compression condition:

Model:

K:

Seed:

Epochs:

Batch size:

Learning rate:

Other changed settings:

Output/result path:

```



Record the exact Git version using:



```bash

git rev-parse HEAD

```



The Git commit is important because it identifies exactly which version of the code produced the result.



\---



\# 19. What Should NOT Be Committed to GitHub



Do \*\*not\*\* commit:



\* FaceForensics++ dataset

\* Celeb-DF dataset

\* Extracted image collections

\* Videos

\* Large model checkpoints

\* Virtual environments

\* Temporary experiment files

\* API keys

\* Passwords

\* Secrets



Large datasets and generated weights should remain outside the public repository unless a separate approved storage/release mechanism is intentionally used.



\---



\# 20. What the Trainer Should Send Back



After training, collect the following.



\## A. Code Version



Run:



```bash

git rev-parse HEAD

```



Send the resulting commit hash.



\---



\## B. Experiment Configuration



Record the exact settings used for each experiment.



\---



\## C. Raw Results



Keep the original result files before converting them into formatted tables.



\---



\## D. Training/Evaluation Logs



Keep logs for:



\* Successful runs

\* Failed runs

\* Unusual runs

\* Runs requiring configuration changes



\---



\## E. Checkpoints



Keep trained model weights separately.



Do not automatically upload large checkpoints to GitHub.



\---



\## F. Final Metrics



Record the actual measured values.



Where applicable:



```text

Mean AUC

Standard Deviation

Other metrics produced by the evaluator

```



\---



\## G. Runtime Information



Record:



```text

GPU:

Training time:

Dataset size:

Number of runs:

Number of seeds:

```



\---



\# 21. Troubleshooting



\## `ModuleNotFoundError`



Make sure the virtual environment is active.



Windows:



```powershell

.venv\\Scripts\\activate

```



Then:



```bash

pip install -r requirements.txt

```



\---



\## CUDA Unavailable



Run:



```bash

python -c "import torch; print(torch.cuda.is\_available())"

```



If the output is:



```text

False

```



check the installed PyTorch/CUDA environment and NVIDIA driver.



\---



\## Dataset Not Found



Check the configured dataset path and verify that:



```text

data/processed/images/

```



contains the expected data.



\---



\## C23 and C40 Results Appear Identical



Check:



1\. C23 data exists.

2\. C40 data exists.

3\. `run\_main.py` passes the requested compression condition.

4\. `loader.py` selects the requested condition.

5\. The experiment is not accidentally reading the same directory for every compression condition.



\---



\## GPU Out-of-Memory Error



Possible adjustments include:



\* Reduce batch size.

\* Reduce the number of workers.

\* Stop other GPU processes.

\* Run fewer experiments simultaneously.



Any change that affects the experimental protocol must be recorded.



\---



\## Reproducibility Problem



Verify that the same:



\* Git commit

\* Dataset

\* Seed

\* K value

\* Compression condition

\* Model configuration

\* Training settings



were used.



\---



\# 22. Research Integrity



The repository contains an experiment implementation, not pre-existing experimental results.



Only report values obtained from actual runs.



Clearly distinguish between:



\* Implementation details

\* Experimental settings

\* Measured results

\* Interpretation of results



Do not fill missing results with expected values.



Do not copy numerical results from another paper and present them as results of this project.



\---



\# 23. Repository Structure



```text

freqlora-clip-deepfake/

│

├── models/

│   └── architectures.py

│

├── data/

│   ├── loader.py

│   └── compression.py

│

├── training/

│   ├── trainer.py

│   └── losses.py

│

├── evaluation/

│   ├── evaluator.py

│   └── metrics.py

│

├── experiments/

│   ├── run\_main.py

│   ├── run\_k\_sweep.py

│   ├── run\_ablations.py

│   ├── run\_cross\_dataset.py

│   └── utils.py

│

├── results/

│   ├── tables.py

│   └── plots.py

│

├── checkpoints/

│

├── requirements.txt

├── setup.py

├── README.md

└── LICENSE

```



\---



\# 24. Final Training Checklist



\## Before Full Training



\* \[ ] Repository cloned/pulled

\* \[ ] Correct Git commit confirmed

\* \[ ] Virtual environment created

\* \[ ] Requirements installed

\* \[ ] PyTorch verified

\* \[ ] GPU checked

\* \[ ] Dataset prepared

\* \[ ] Dataset paths verified

\* \[ ] C23 verified

\* \[ ] C40 verified

\* \[ ] Sanity test passed

\* \[ ] Main experiment configuration checked



\## After Training



\* \[ ] M1 results saved

\* \[ ] M2 results saved

\* \[ ] M3 results saved

\* \[ ] M4 results saved

\* \[ ] K-sweep results saved

\* \[ ] Ablation results saved

\* \[ ] Cross-dataset results saved

\* \[ ] Raw logs preserved

\* \[ ] Checkpoints backed up

\* \[ ] Git commit recorded

\* \[ ] Actual metrics verified

\* \[ ] Tables generated

\* \[ ] Plots generated

\* \[ ] No fabricated values added



\---



\# 25. Important Note for the Research Team



The person performing the training should \*\*not change the experimental setup just to obtain better-looking results\*\*.



If a problem requires changing:



\* dataset preprocessing

\* model configuration

\* K

\* seed

\* learning rate

\* batch size

\* number of epochs

\* compression condition

\* evaluation protocol



the change should be recorded clearly.



This keeps the experiments reproducible and makes the final research results defensible.



