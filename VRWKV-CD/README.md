# VRWKV-CD

Official-style implementation of a dual-branch hyperspectral image change
detection network. The spatial branch models bidirectional temporal-spatial
cross sequences with VRWKV; the centre-spectrum branch performs adaptive
multi-scale spectral comparison. Their calibrated class probabilities are
combined using a fixed dataset-level weight.

## Architecture

1. **Spatial VRWKV branch.** T1 and T2 patches are projected to 128 channels.
   T1→T2 and T2→T1 features are interleaved along horizontal and vertical
   axes, processed by a depth-3 VRWKV encoder, restored to their original
   coordinates, and fused. An absolute-change residual gate refines the
   resulting spatial representation.
2. **Centre-spectrum branch.** Only the aligned centre spectra are used. The
   absolute temporal difference and temporal mean are encoded at three
   spectral scales, adaptively fused, projected along the spectral axis to
   length 32, and mapped to a 64-dimensional representation.
3. **Probability fusion.** The spatial, spectral, and fused predictions each
   receive cross-entropy supervision. Inference combines the two branch
   probabilities with a fixed spectral weight specified in
   `configs/final.json`.

The final configuration does not use Mamba, Sobel modulation, a physics head,
VAT, prototype loss, label smoothing, or training augmentation.

## Installation

```bash
conda create -n vrwkv-cd python=3.10 -y
conda activate vrwkv-cd
pip install -r requirements.txt
```

The custom WKV CUDA extension is compiled on first use. Set `CUDA_HOME` when
CUDA is not installed at `/usr/local/cuda`.

## Data

Set `HSICD_DATA_ROOT` to the dataset root. Expected files are:

```text
$HSICD_DATA_ROOT/
├── River/{river_before.mat,river_after.mat,groundtruth.mat}
├── farmland2/{farm06.mat,farm07.mat,label.mat}
├── BayArea/{Bay_Area_2013.mat,Bay_Area_2015.mat,bayArea_gtChanges2.mat}
└── Hermiston390_CITIUS/Hermiston/
    ├── hermiston2004.mat
    ├── hermiston2007.mat
    └── rdChangesHermiston_5classes.mat
```

Dataset aliases used by the scripts are `river`, `yancheng`, `bayarea`, and
`hermiston390`. Dataset files are not distributed with this repository.

## Training

```bash
export HSICD_DATA_ROOT=/path/to/data
export OUTPUT_ROOT=./outputs

# One seed
./scripts/run_seed.sh 0 river 0 val50_of_1pct

# Seeds 0--9
./scripts/run_10seeds.sh 0 river val50_of_1pct
```

The default protocol reserves half of the class-wise 1% label budget for
validation and selects the checkpoint with the best validation OA. To
reproduce the diagnostic 1%-training/test-peak protocol, replace
`val50_of_1pct` with `test_peak_1pct`.

Input normalization is fitted using training patches only. The trainer reports
both the ordinary test complement and a strict spatially disjoint evaluation
that removes test centres within Chebyshev distance 6 of a supervised centre.

## Reproducibility

- seeds: 0--9;
- optimizer: AdamW, learning rate `6e-4`, weight decay `1e-5`;
- scheduler: cosine decay to 1% of the initial learning rate;
- 100 epochs, evaluation every 10 epochs;
- all reported metrics are computed from the changed class: precision, recall,
  F1, and IoU; OA and Kappa use the complete binary confusion matrix.

See `configs/final.json` for dataset-specific patch sizes, encoder sharing, and
fusion weights.

Summarize completed seeds with:

```bash
python scripts/summarize.py outputs/results/final/val50_of_1pct/river \
  --output river_summary.md
```

The released environment was tested with PyTorch 2.1.0, CUDA 11.8,
NumPy 1.24.4, SciPy 1.10.1, scikit-learn 1.3.2, einops 0.8.1, and
mamba-ssm 1.1.3.

## Citation

Citation information will be added after publication.
