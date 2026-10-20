# Quantized Spiking Neural Networks for Gravitational-Wave Glitch Classification

Does a quantised spiking neural network base its decisions on the same evidence
as its full-precision counterpart, and at what energy cost?

The classifier assigns transient noise ("glitches") in LIGO data to four
morphological classes. The Spike Activation Map (SAM) is used to compare what
the network attends to before and after compression, with weights and membrane
potential quantised separately and then together, at 8, 4 and 2 bits.

The thesis (LaTeX sources and compiled PDF) is in `thesis_report/thesis/`;
Appendix C of the PDF maps every table and figure to the script that produces it.

## Setup

Python >= 3.12. From the repository root:

```bash
pip install -e .
pip install -r requirements.txt
```

To reproduce the exact environment of the thesis (torch 2.6.0 + CUDA 12.4,
snnTorch 1.0.0), install the pinned versions instead of `requirements.txt`:

```bash
pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu124
```

`python -m model.quantization` checks the quantizer properties on whatever
version is installed and prints `all properties hold`.

Every script is run **from the repository root, as a module**, because paths
such as `config/baseline_manifest.json` are relative to it:

```bash
python -m scripts.train --seed 1234
python -m xai.divergence_metrics --no-ssim
```

## What is and is not in the repository

| Included | Not included |
| --- | --- |
| Code, `config/` (manifest, hyperparameters, membrane range) | Raw metadata and spectrograms (`data/raw/`, `data/processed/`) |
| `data/split_assignment.csv`, the frozen partition | Trained checkpoints (`runs/<tag>/*.pt`) |
| `data/corrupted_files.txt`, the two images lost at source | Optuna databases and W&B logs |
| `results/`: every JSON and CSV behind the tables of the thesis | |

The data are downloaded by steps 1–3 below from Zenodo record
[5649212](https://doi.org/10.5281/zenodo.5649212); the corpus is 9 566 images
across four classes. Labels are the classifications of the Gravity Spy
convolutional model, not physical ground truth, so reported accuracy is
agreement with that reference classifier.

Checkpoints are written to `runs/<tag>/` (`best_model.pt`, `final_model.pt`,
`run.json`). Every step from the FP32 reference onwards needs them, so the
analyses can only be re-run after the training steps; the numbers in `results/`
can be inspected without them.

## Pipeline

### Full-precision stages

| # | Module | What it does | Output |
| --- | --- | --- | --- |
| 1 | `scripts.dataset_downloader` | Fetch the eight Gravity Spy metadata CSVs from Zenodo | `data/raw/` |
| 2 | `scripts.data_prep` | Exploratory analysis, class and feature selection, balancing | `data/processed/selected_dataset.csv` |
| 3 | `scripts.image_downloader` | Download one Omega-scan spectrogram per retained event | `data/processed/<class>/` |
| 4 | `scripts.freeze_split` | Write the 70/15/15 partition to disk, once | `data/split_assignment.csv` |
| 5 | `scripts.tabular_baseline` | Logistic regression, random forest, gradient boosting on the four descriptors | `results/tabular_baseline.json` |
| 6 | `scripts.resolution_study` | Three resolutions × three seeds | `runs/resolution_study.csv`, copied to `results/` |
| 7 | `scripts.dead_neurons_by_resolution --out results/dead_neurons.json` | Dead and saturated fractions by resolution | `results/dead_neurons.json` |
| 8 | `scripts.tune` | Optuna search (v4 free, v5 shift-implementable β and T ≤ 8) | `config/best_hyperparams*.json` |
| 9 | `scripts.fp32_reference` | Three-seed reference | `config/baseline_manifest.json` |
| 10 | `scripts.verify_manifest` | Check the partition digest and the checkpoints named in the manifest | — |
| 11 | `scripts.dead_neuron_cutoff_sweep` | Sweep of the dead and saturated cutoffs | `results/tables/dead_neuron_sweep.csv` |
| 12 | `scripts.diagnostics_figures` | Confusion matrix, firing-rate distributions, CV-ISI | `results/Figures/` |
| 13 | `scripts.sam_visualization` | One SAM panel per time step, single sample | figure |
| 14 | `xai.sam_class_comparison`, `xai.sam_layer_comparison` | SAM statistics per class and per layer | figures |
| 15 | `xai.gamma_sensitivity_analysis` | Fixes the SAM decay rate γ | `results/gamma_sensitivity.json` |
| 16 | `xai.deletion_metric` | Faithfulness of SAM against a random baseline | `results/Figures/deletion_metric.pdf` |
| 17 | `xai.noise_floor` | FP32-vs-FP32 divergence across seed pairs | `results/noise_floor.json` |
| 18 | `xai.perturbation_floor` | Divergence under negligible weight noise | `results/perturbation_floor.json` |
| 19 | `scripts.snr_axis_check` | Verifies that SNR translates into image intensity | console |

Steps 6–8 sweep the resolution through the environment variable
`SNN_INPUT_SIZE`; from step 9 onwards every script reads it from the manifest.

### Quantisation study

| # | Module | What it does | Output |
| --- | --- | --- | --- |
| 20 | `scripts.membrane_range` | Range of the membrane potential per layer | `config/membrane.json` |
| 21 | `scripts.test_clipping` | Clipping study on the FP32 network | `results/clipping_test.json` |
| 22 | `model.quantization` | Self-checks of the quantisers (run as a module) | console |
| 23 | `scripts.ptq_check` | PTQ gate and fold/reload round trip | `results/ptq_check.json` |
| 24 | `scripts.grid_runner` | The 30 fine-tuning runs (calls `scripts.train_quantised`) | `runs/qat_*`, `results/grid_summary.json` |
| 25 | `scripts.evaluate_model` | One validation pass per model: accuracy, rates, dead/saturated, CV-ISI | `results/grid/` |
| 26 | `scripts.finetuning_anchor` | The control arm after its last epoch | `results/finetuning_anchor.json` |
| 27 | `xai.divergence_metrics --no-ssim` | SAM divergence of every model from its own parent | `results/divergence.json` |
| 28 | `xai.cross_faithfulness` | Deletion metric on the 2-bit membrane models | `results/cross_faithfulness.json` |
| 29 | `xai.stratify` | IoU by within-class SNR tercile | `results/stratified.json` |
| 30 | `scripts.grid_analysis` | Firing-rate EMD and disagreement subset | `results/grid_analysis.json` |
| 31 | `scripts.per_image_cv` | CV-ISI inside each image | `results/per_image_cv.json` |
| 32 | `xai.energy` | SOPs, energy and memory per model | `results/energy.json` |
| 33 | `xai.energy --rates final --baseline qat_wxux_{input_size}px_seed{seed} --out results/energy_final.json` | Energy against the control | `results/energy_final.json` |
| 34 | `xai.sam_qat_comparison --run <tag>` | SAM maps side by side, top-k sweep | figures, `results/tables/` |
| 35 | `scripts.figures.result_figures` | Figures and tables of the grid | `results/tables/`, figures |
| 36 | `scripts.figures.grid_headline_figure` | Summary figure of the grid | `thesis_report/thesis/figures/grid_headline.pdf` |
| 37 | `scripts.architecture_figure`, `scripts.figures.*` | Method figures | figures |
| 38 | `scripts.robustness_checks all` | Readout ties, per-image CV of the controls, bootstrap intervals of IoU and deletion gaps | `results/robustness/` |
| 39 | `scripts.evaluate_test` | **Opens the test partition. Once.** | `results/test/`, `results/test_opened.json` |

The test partition was opened once, on 7 October 2026, for all 33 models; the
script refuses a second opening unless forced. Step 38 only reads validation
data and can be run at any time; it checks that it reproduces the stored means
before reporting intervals.

## Reproducibility switches

Both are off by default, which is how every reported run was produced.

| Variable | Effect |
| --- | --- |
| `SNN_DETERMINISTIC=1` | `scripts.train` and `scripts.train_quantised` select deterministic cuDNN kernels |
| `OPTUNA_SEED=<int>` | `scripts.tune` seeds its TPE sampler |

## Layout

| Path | Contents |
| --- | --- |
| `model/` | `snn_model.py`, the architecture; `quantization.py`, the quantisers |
| `scripts/` | Data pipeline, training, fine-tuning, diagnostics, test evaluation, figures |
| `xai/` | SAM, its metrics, their calibration, faithfulness, energy |
| `config/` | Frozen manifest, hyperparameters, membrane range |
| `data/` | `split_assignment.csv` and `corrupted_files.txt` |
| `results/` | The JSON and CSV behind every table, and the figures |
| `thesis_report/` | LaTeX sources and the compiled PDF |

## Library modules

| Module | Role |
| --- | --- |
| `model/snn_model.py` | `GWGlitchSNN`: four conv+LIF blocks, rate-coded readout |
| `model/quantization.py` | Per-channel weight quantiser, membrane quantiser on [0, 2], fold and removal |
| `scripts/dataloader.py` | Transforms, frozen-split loading, `build_dataloaders` |
| `scripts/training_utils.py` | Encoding, training loop, validation, firing rates |
| `scripts/train.py` | `run_training`, used by the resolution study and the reference |
| `scripts/train_quantised.py` | `run_qat`, quantisation-aware fine-tuning from the parent of the same seed |
| `scripts/per_sample.py` | Per-sample output and metadata joins |
| `xai/sam.py` | TSCS, NCS, SAM, temporal centre of mass, model loading |
| `xai/sam_metrics.py` | Sampling, Pearson, top-k IoU, nan-aware aggregation |
