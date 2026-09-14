# Quantized Spiking Neural Networks for Gravitational-Wave Glitch Classification


Does a quantised spiking neural network base its decisions on the same evidence
as its full-precision counterpart?

The classifier assigns transient noise
("glitches") in LIGO data to four morphological classes, and the Spike
Activation Map is used to compare what the network attends to before and after
compression.

## Setup

```bash
pip install -e .
pip install -r requirements.txt
```

## Pipeline


| # | Script | What it does |
| --- | --- | --- |
| 1 | `scripts/dataset_downloader.py` | Fetch the eight Gravity Spy metadata CSVs from Zenodo |
| 2 | `scripts/data_prep.py` | Exploratory analysis, class and feature selection, balancing |
| 3 | `scripts/image_downloader.py` | Download one Omega-scan spectrogram per retained event |
| 4 | `scripts/freeze_split.py` | Write the 70/15/15 partition to disk, once |
| 5 | `scripts/resolution_study.py` | Three resolutions × three seeds; fixes the working resolution |
| 6 | `scripts/dead_neurons_check.py` | Per-neuron firing rates; the diagnostic that decides the resolution |
| 7 | `scripts/tune.py` | Optuna search under shift-implementable β and T ≤ 8 |
| 8 | `scripts/fp32_reference.py` | Three-seed reference; writes `config/baseline_manifest.json` |
| 9 | `scripts/physical_validation.py` | Confusion matrix, firing-rate distributions, CV-ISI |
| 10 | `xai/gamma_sensitivity_analysis.py` | Fixes the SAM decay rate γ |
| 11 | `scripts/sam_visualization.py` | One SAM panel per time step, single sample |
| 12 | `xai/sam_class_comparison.py` | Mean, std and CV maps per class |
| 13 | `xai/sam_layer_comparison.py` | The same statistics across all four hidden layers |
| 14 | `xai/deletion_metric.py` | Faithfulness of SAM against a random baseline |
| 15 | `xai/noise_floor.py` | FP32-vs-FP32 divergence across seed pairs |
| 16 | `xai/noise_floor_selfcheck.py` | Identity, determinism and reproducibility checks |
| 17 | `xai/perturbation_floor.py` | Divergence under negligible weight noise |
| 18 | `scripts/snr_axis_check.py` | Verifies SNR translates into image intensity |
| 19 | `scripts/evaluate_test.py` | **Opens the test partition. Once.** |

Resolution is passed through `SNN_INPUT_SIZE` for steps 5–7, which sweep over
it; from step 8 onwards every script reads it from the manifest.

## Layout

| Path | Contents |
| --- | --- |
| `model/` | `snn_model.py` — the architecture, and nothing else |
| `scripts/` | Data pipeline, training, diagnostics, entry points |
| `xai/` | SAM, its metrics, and the calibration of those metrics |
| `config/` | Frozen manifest and hyperparameters |
| `data/` | Only `split_assignment.csv`, the frozen partition |
| `results/` | Figures and the JSON behind every table in the report |
| `thesis_report/` | LaTeX sources and the compiled PDF |


## Library modules

| Module | Role |
| --- | --- |
| `model/snn_model.py` | `GWGlitchSNN`: four conv+LIF blocks, rate-coded readout |
| `scripts/dataloader.py` | Transforms, frozen-split loading, `build_dataloaders` |
| `scripts/training_utils.py` | Encoding, training loop, validation, firing rates |
| `scripts/train.py` | `run_training`, used by the resolution study and the reference |
| `scripts/per_sample.py` | Per-sample output and metadata joins |
| `xai/sam.py` | TSCS, NCS, SAM, temporal centre of mass, model loading |
| `xai/sam_metrics.py` | Sampling, Pearson, top-k IoU, nan-aware aggregation |

## Data

Not included. The scripts download it from Zenodo record
[5649212](https://doi.org/10.5281/zenodo.5649212); the corpus is 9 566 images
across four classes. The frozen partition **is** version-controlled, and the
manifest records its SHA-256 digest: without it the results cannot be
reproduced.

Labels are the classifications of the Gravity Spy convolutional model, not
physical ground truth, so reported accuracy is agreement with that reference
classifier.