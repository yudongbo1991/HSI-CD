# VRWKV-CD

VRWKV-CD is a dual-branch network for hyperspectral image change detection. It combines local spatial context with center-pixel spectral changes and fuses the predictions of the two branches at the probability level.

## Method Overview

### 1. Spatial Context Branch

The spatial branch takes paired hyperspectral patches centered at the same spatial location in the two temporal images:

1. Two `1×1` convolutions project the spectral channels of T1 and T2 to 128 dimensions. River shares one projection between T1 and T2. The other datasets use two independently optimized projections initialized with identical weights.
2. T1 and T2 pixels are interleaved along the horizontal direction and the true vertical direction to form two cross-temporal spatial sequences.
3. The two sequences are processed by a shared three-layer VRWKV encoder for bidirectional WKV context aggregation.
4. Each sequence is restored to its original spatial coordinates. T1 and T2 position features are fused with a `1×1` convolution, and the horizontal and vertical features are averaged.
5. Absolute change evidence is computed from the projected T1 and T2 features and injected into the VRWKV representation through a learnable residual gate.
6. The spatial representation is flattened and classified using `BatchNorm → LeakyReLU → Dropout(0.05) → Linear`.

The VRWKV decay parameters use a signed symmetric initialization whose magnitudes are logarithmically distributed from 0.15 to 4.0. The retained VRWKV operator also includes its native edge-aware receptance mechanism.

### 2. Center-Spectrum Branch

This branch uses only the paired spectra at the center pixel of each patch:

1. It constructs the absolute difference `|T2-T1|` and temporal mean `(T1+T2)/2`.
2. Three spectral encoders extract local, regional, and long-range patterns using a kernel-3 convolution, a kernel-7 convolution, and a dilated kernel-3 convolution with dilation 4.
3. A lightweight scale selector predicts three normalized weights from each sample's scale summaries and adaptively fuses the three spectral representations.
4. A learnable linear projection along the spectral axis reduces the spectral length to 32.
5. The fused representation is mapped to a 64-dimensional feature and classified into changed or unchanged.

### 3. Dual-Branch Fusion and Training Objective

The two branches independently produce class probabilities. Their final prediction is

```text
P = spatial_weight × P_spatial + spectral_weight × P_spectral
spatial_weight = 1 - spectral_weight
```

The model is trained with three equally weighted cross-entropy losses:

```text
L = L_fused + L_spatial + L_spectral
```

## Experimental Protocol

- A class-wise 1% label budget is sampled independently from each class.
- Within this budget, each class is split equally into training and validation samples.
- Normalization statistics are estimated only from patches centered on training samples. Validation and test samples are never used to fit the normalization parameters.
- Validation is performed every 10 epochs. The checkpoint with the highest validation OA is selected; validation loss is used as the tie-breaker.
- The selected model is evaluated only once after training.
- Two test scopes are reported:
  - `loose99`: all labeled pixels outside the 1% label budget;
  - `strict7`: a spatially disjoint subset of `loose99`, excluding pixels within a Chebyshev distance of 6 from any pixel in the 1% label budget.
- Seeds 0–9 are used by default, and results are reported as mean ± population standard deviation.

## Default Configuration

| Dataset | Patch size | Spatial weight | Spectral weight | T1/T2 spatial projection |
|---|---:|---:|---:|---|
| River | 7×7 | 0.5 | 0.5 | Shared |
| Yancheng | 7×7 | 0.4 | 0.6 | Independent, identical initialization |
| BayArea | 7×7 | 0.6 | 0.4 | Independent, identical initialization |
| Hermiston390 | 5×5 | 0.5 | 0.5 | Independent, identical initialization |

Other default training settings are listed below.

| Setting | Value |
|---|---:|
| Training epochs | 100 |
| Validation interval | 10 |
| Training batch size | 64 |
| Evaluation batch size | 128 |
| Optimizer | AdamW with AMSGrad |
| Initial learning rate | `6e-4` |
| Weight decay | `1e-5` |
| Learning-rate scheduler | CosineAnnealingLR |
| Minimum learning-rate ratio | 0.01 |
| Gradient clipping | 1.0 |
| Spatial input dimension | 128 |
| Spatial output dimension | 64 |
| VRWKV depth | 3 |
| Spectral encoder width | 16 |
| Spectral projection length | 32 |
| Spectral representation dimension | 64 |

The complete default configuration is also recorded in [`configs/final.json`](configs/final.json). The patch size, fusion weight, and spatial projection sharing mode can be overridden from the command line.

## Installation

Conda is recommended:

```bash
conda env create -f environment.yml
conda activate vrwkv-cd
```

The WKV CUDA extension in `core/cuda_new/` is compiled when the model is imported for the first time. The following components are therefore required:

- an NVIDIA GPU with a compatible driver;
- a CUDA Toolkit installation with `nvcc`;
- a C++ compiler and Ninja.

If CUDA cannot be detected automatically, set `CUDA_HOME` explicitly:

```bash
export CUDA_HOME=/path/to/cuda
```

## Dataset Organization

Place the datasets under the repository's `data/` directory, or set `HSICD_DATA_ROOT` to an external dataset directory:

```text
data/
├── River/
│   ├── river_before.mat
│   ├── river_after.mat
│   └── groundtruth.mat
├── farmland2/
│   ├── farm06.mat
│   ├── farm07.mat
│   └── label.mat
├── BayArea/
│   ├── Bay_Area_2013.mat
│   ├── Bay_Area_2015.mat
│   └── bayArea_gtChanges2.mat
└── Hermiston390_CITIUS/
    └── Hermiston/
        ├── hermiston2004.mat
        ├── hermiston2007.mat
        └── rdChangesHermiston_5classes.mat
```

The dataset identifiers accepted by the training program are `river`, `yancheng`, `bayarea`, and `hermiston390`.

## Running the Code

### One Seed

```bash
export HSICD_DATA_ROOT=/path/to/data
export OUTPUT_ROOT=./outputs

./scripts/run_seed.sh 0 river 0
```

The three positional arguments are the GPU index, dataset name, and random seed.

The Python entry point can also be used directly:

```bash
python train.py \
  --dataset river \
  --seed 0 \
  --gpu 0 \
  --epochs 100 \
  --eval-every 10 \
  --batch-size 64 \
  --test-batch-size 128 \
  --output outputs/river/seed_0
```

### Ten Seeds

```bash
./scripts/run_10seeds.sh 0 river
```

This command runs seeds 0–9 sequentially.

### Custom Patch Size and Fusion Weight

The following example uses a `9×9` patch and assigns a weight of 0.45 to the spectral branch:

```bash
python train.py \
  --dataset river \
  --seed 0 \
  --gpu 0 \
  --patch 9 \
  --spectral-weight 0.45 \
  --output outputs/river_patch9/seed_0
```

The spatial weight is automatically set to `1 - spectral_weight`. T1/T2 spatial projections can be configured with either of the following flags:

```text
--share-spatial-projection
--no-share-spatial-projection
```

## Output Files

Each seed produces the following files:

```text
seed_0/
├── train.log
├── result.json
├── predictions.npz
├── loose99_error_map.png
└── strict7_error_map.png
```

- `train.log` records the training loss, training OA, and periodic validation OA.
- `result.json` contains the configuration, selected epoch, sample counts, training history, and metrics for both test scopes.
- `predictions.npz` stores sample coordinates, labels, and predictions for reproducible analysis and visualization.
- In the error maps, white denotes correctly detected changed pixels, black denotes correctly detected unchanged pixels, red denotes missed changes, blue denotes false alarms, and gray denotes unevaluated pixels.

## Summarizing Ten-Seed Results

```bash
python scripts/summarize.py outputs/river \
  --output outputs/river_summary.md
```

The summary contains OA, Kappa, Precision, Recall, F1, and IoU. Precision, Recall, F1, and IoU are computed for the changed class.

## Repository Structure

```text
VRWKV-CD/
├── core/
│   ├── model.py
│   ├── hi_rwkv.py
│   ├── vrwkv.py
│   └── cuda_new/
├── configs/final.json
├── scripts/
│   ├── run_seed.sh
│   ├── run_10seeds.sh
│   └── summarize.py
├── data.py
├── metrics.py
├── train.py
├── environment.yml
└── README.md
```
