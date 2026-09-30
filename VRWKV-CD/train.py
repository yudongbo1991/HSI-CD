#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path


def _gpu_from_argv():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu", default="0")
    known, _ = parser.parse_known_args()
    return known.gpu


os.environ["CUDA_VISIBLE_DEVICES"] = _gpu_from_argv()
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-12.1")
# Every currently available L40S/RTX4090 device is compute capability 8.9.
# A single target makes concurrent jobs share one extension cache artifact.
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"
# Do not inherit hidden RWKV experiment settings from the launching shell.
os.environ["RWKV_DECAY_INIT"] = "original"
os.environ["RWKV_DECAY_PARAM"] = "direct"
os.environ["RWKV_DECAY_MIN"] = "0.15"
os.environ["RWKV_DECAY_MAX"] = "4.0"

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from core import CleanVRWKVCD
from data import (DATASET_NAMES, LazyPairDataset, load_benchmark,
                                 reflect_pad)
from metrics import binary_metrics, format_metrics
from physics import TrainOnlyPhysicsCalibrator


class FocalCrossEntropy(torch.nn.Module):
    """Smoothly emphasize uncertain labelled samples without hard mining."""
    def __init__(self, weight, label_smoothing=0.0, gamma=0.5):
        super().__init__()
        self.register_buffer("class_weight", weight)
        self.label_smoothing = float(label_smoothing)
        self.gamma = float(gamma)

    def forward(self, logits, target):
        per_sample = F.cross_entropy(
            logits, target, weight=self.class_weight, reduction="none",
            label_smoothing=self.label_smoothing)
        probability = F.softmax(logits, dim=1).gather(
            1, target[:, None]).squeeze(1)
        # For gamma < 1, d(x**gamma)/dx is singular at x=0. Predictions can
        # become exactly one in float32, so bound the modulation before pow.
        modulation = (1.0 - probability).clamp_min(1e-4).pow(self.gamma)
        return (modulation * per_sample).mean()


def arguments():
    parser = argparse.ArgumentParser("Clean VRWKV hyperspectral change detection")
    parser.add_argument("--variant", choices=sorted(CleanVRWKVCD.VARIANTS),
                        default=CleanVRWKVCD.FINAL_VARIANT)
    parser.add_argument("--local-only-views", action="store_true",
                        help="replace the 9x9 pooled context with the same centred 5x5 crop")
    parser.add_argument("--local-patch", type=int, default=5,
                        help="centred local-view size; default 5 preserves the released base")
    parser.add_argument("--disable-context-modeling", action="store_true",
                        help="bypass the context view, context gate and context auxiliary head")
    parser.add_argument("--spectral-spatial-order",
                        choices=("spectral_first", "spatial_first", "parallel"),
                        default="spectral_first")
    parser.add_argument("--spectral-groups", type=int, default=16)
    parser.add_argument("--mamba-state", type=int, default=16)
    parser.add_argument("--mamba-intra-state", type=int, default=0,
                        help="intra-group state size; 0 reuses --mamba-state")
    parser.add_argument("--mamba-inter-state", type=int, default=0,
                        help="inter-group state size; 0 reuses --mamba-state")
    parser.add_argument("--mamba-expand", type=int, default=2)
    parser.add_argument("--mamba-depth", type=int, default=1,
                        help="number of stacked grouped Mamba stages")
    parser.add_argument("--vrwkv-depth", type=int, default=2)
    parser.add_argument("--classifier-dropout", type=float, default=0.1)
    parser.add_argument("--spatial-input-dim", type=int, default=128)
    parser.add_argument("--spatial-feature-dim", type=int, default=64)
    parser.add_argument("--spatial-unshared-input-projection",
                        action="store_true")
    parser.add_argument("--change-residual-mode", choices=("gated", "direct"),
                        default="gated")
    parser.add_argument("--spatial-channel-attention",
                        choices=("none", "pre", "post"), default="none")
    parser.add_argument("--spatial-classifier-mode",
                        choices=("flatten", "global_avg", "center", "center3",
                                 "gaussian"),
                        default="flatten")
    parser.add_argument("--spectral-inter-only", action="store_true")
    parser.add_argument("--vrwkv-edge-mode",
                        choices=("legacy", "normalized_context", "off"),
                        default="legacy")
    parser.add_argument("--dataset", choices=DATASET_NAMES, default="hermiston")
    parser.add_argument("--train-fraction", type=float, default=0.01,
                        help="strict class-wise labelled training fraction")
    parser.add_argument("--input-normalization",
                        choices=("scene_joint", "train_patches"),
                        default="scene_joint")
    parser.add_argument(
        "--selection-protocol",
        choices=("test_peak_1pct", "val10_of_1pct", "val50_of_1pct"),
        default="test_peak_1pct",
        help=("epoch-selection rule. Validation protocols split only the "
              "fixed labelled pool; the untouched test set is unchanged"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--test-freq", type=int, default=10)
    parser.add_argument(
        "--test-at-round-multiples", action="store_true",
        help="evaluate at test_freq, 2*test_freq, ... instead of epoch 1, 1+test_freq, ...")
    parser.add_argument("--patch", type=int, default=9)
    parser.add_argument("--context-pool-mode",
                        choices=("average", "hierarchical_context",
                                 "two_scale_middle", "two_scale_global",
                                 "two_scale_global_aux",
                                 "aligned_multikernel",
                                 "aligned_reliability_multikernel",
                                 "aligned_reliability_sharp",
                                 "paired_reliability_multikernel",
                                 "temporal_reliability_multikernel",
                                 "aligned_temporal_reliability",
                                 "edge_aware",
                                 "edge_residual"),
                        default="average",
                        help="context compression before the shared backbone")
    parser.add_argument("--context-temporal-weight", type=float, default=0.5,
                        help="temporal-change preservation term in context reliability")
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--test-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--rwkv-decay-min", type=float, default=0.15,
                        help="minimum signed-symmetric VRWKV decay magnitude")
    parser.add_argument("--rwkv-decay-max", type=float, default=4.0,
                        help="maximum signed-symmetric VRWKV decay magnitude")
    parser.add_argument("--rwkv-lr-multiplier", type=float, default=1.0,
                        help="learning-rate multiplier for spatial_decay parameters")
    parser.add_argument("--lr-min-ratio", type=float, default=0.01,
                        help="Cosine scheduler minimum LR as a fraction of base LR")
    parser.add_argument("--lr-scheduler",
                        choices=("cosine", "constant", "step", "multistep"),
                        default="cosine",
                        help="learning-rate schedule; cosine preserves the base behavior")
    parser.add_argument("--lr-step-size", type=int, default=30,
                        help="epoch interval used by the step scheduler")
    parser.add_argument("--lr-gamma", type=float, default=0.1,
                        help="multiplicative decay used by step/multistep schedules")
    parser.add_argument("--lr-milestones", type=int, nargs="*", default=(35, 70),
                        help="decay epochs used by the multistep scheduler")
    parser.add_argument("--disable-amsgrad", action="store_true",
                        help="Use standard AdamW instead of AMSGrad")
    parser.add_argument("--label-smoothing", type=float, default=-1.0,
                        help="negative keeps the variant default (0.02 for main variants)")
    parser.add_argument("--focal-gamma", type=float, default=0.0,
                        help="positive enables smooth hard-sample reweighting")
    parser.add_argument("--change-class-weight", type=float, default=1.0,
                        help="relative CE weight for changed class (label zero)")
    parser.add_argument("--coarse-aux-weight", type=float, default=0.3,
                        help="CE weight of the coarse class-state head")
    parser.add_argument("--coarse-ablation-mode",
                        choices=("normal", "context_vectors", "no_refinement"),
                        default="normal",
                        help="semantic coarse-stage ablation used in River tests")
    parser.add_argument("--context-aux-weight", type=float, default=0.10,
                        help="total reliability-weighted CE weight of the context head")
    parser.add_argument("--embedded-physics-aux-weight", type=float, default=0.02,
                        help="total CE weight of the embedded physical head")
    parser.add_argument("--learned-difference-aux-weight", type=float, default=0.02,
                        help="CE weight of the learned sample-difference head; kept separate from the physical head")
    parser.add_argument("--scene-codebook-size", type=int, default=0,
                        help="prototypes per class; zero disables the learned scene prior")
    parser.add_argument("--scene-prior-dim", type=int, default=32)
    parser.add_argument("--scene-reliability-gate", action="store_true",
                        help="attenuate prototype modulation by confidence and coverage")
    parser.add_argument("--pixel-spectral-groups", type=int, default=0,
                        help="groups in the learned centre-spectrum branch; zero disables")
    parser.add_argument("--pixel-spectral-width", type=int, default=16)
    parser.add_argument("--pixel-spectral-dim", type=int, default=64)
    parser.add_argument("--pixel-spectral-linear-length", type=int, default=0)
    parser.add_argument("--pixel-spectral-kernel", type=int, choices=(3, 5), default=3)
    parser.add_argument("--pixel-spectral-design",
                        choices=("symmetric", "learned_descriptor",
                                 "learned_descriptor_prototype"),
                        default="symmetric")
    parser.add_argument("--spectral-only-classifier", action="store_true",
                        help="bypass the context backbone and classify center spectra only")
    parser.add_argument("--spectral-difference-only", action="store_true",
                        help="classify the raw signed center-spectrum difference linearly")
    parser.add_argument("--patch-difference-only", action="store_true",
                        help="classify the flattened signed 5x5 patch difference linearly")
    parser.add_argument("--pixel-fusion-alpha", type=float, default=0.5)
    parser.add_argument("--test-fusion-alpha-override", type=float,
                        default=None,
                        help="spectral probability weight used only for final test")
    parser.add_argument("--test-exclusion-radius", type=int, default=-1,
                        help=("if nonnegative, evaluate only test centres whose "
                              "Chebyshev distance from every labelled-pool "
                              "centre exceeds this radius"))
    parser.add_argument("--pixel-adaptive-fusion", action="store_true",
                        help=("learn a soft scene preference and bounded "
                              "per-sample confidence gate end to end"))
    parser.add_argument("--pixel-independent-fusion", action="store_true",
                        help=("learn independent positive main and spectral "
                              "evidence scales in logit space"))
    parser.add_argument("--pixel-dual-reliability-fusion", action="store_true",
                        help=("predict two independent per-sample branch "
                              "reliabilities after logit normalization"))
    parser.add_argument("--pixel-complementarity-fusion", action="store_true",
                        help=("compute one deterministic centre-context "
                              "complementarity weight per patch during both "
                              "training and inference"))
    parser.add_argument("--pixel-complementarity-v2-fusion", action="store_true",
                        help=("use threshold-free direction, neighbourhood "
                              "coherence and non-redundancy evidence per patch"))
    parser.add_argument("--pixel-complementarity-v3-fusion", action="store_true",
                        help="threshold-free coherent-novelty pixel fusion")
    parser.add_argument("--pixel-complementarity-v4-fusion", action="store_true",
                        help="four-region joint physical evidence fusion")
    parser.add_argument("--pixel-complementarity-v5-fusion", action="store_true",
                        help="scene-consensus four-region evidence fusion")
    parser.add_argument("--pixel-complementarity-v6-fusion", action="store_true",
                        help="balanced coherent-novelty scene evidence")
    parser.add_argument("--pixel-complementarity-v7-fusion", action="store_true",
                        help="simple consensus-or-novelty scene fusion")
    parser.add_argument("--pixel-complementarity-v8-fusion", action="store_true",
                        help="simple consensus and moderate-novelty fusion")
    parser.add_argument("--pixel-complementarity-v9-fusion", action="store_true",
                        help="two-bandpass physical scene fusion")
    parser.add_argument("--pixel-complementarity-v10-fusion", action="store_true",
                        help="robust scene-range complementarity fusion")
    parser.add_argument("--pixel-complementarity-v11-fusion", action="store_true",
                        help="unified consensus-novelty-range fusion")
    parser.add_argument("--pixel-batch-scene-fusion", action="store_true",
                        help="use each mini-batch as a scene consistently in train/eval")
    parser.add_argument("--pixel-batch-full-v11-fusion", action="store_true",
                        help="apply complete v11 C/N/R rule inside every batch")
    parser.add_argument("--pixel-complementarity-v12-fusion", action="store_true",
                        help="redundancy-aware consensus-novelty-range fusion")
    parser.add_argument("--pixel-complementarity-v13-fusion", action="store_true",
                        help="tempered-evidence scene fusion")
    parser.add_argument("--pixel-complementarity-v14-fusion", action="store_true",
                        help="inter-group complementarity scene fusion")
    parser.add_argument("--pixel-two-statistic-fusion", action="store_true",
                        help="per-patch cosine/residual physical fusion")
    parser.add_argument("--pixel-geomean-fusion", action="store_true",
                        help="per-patch geometric-mean cosine/residual fusion")
    parser.add_argument("--pixel-decomposed-context-fusion", action="store_true",
                        help="direction/coherence/amplitude context fusion")
    parser.add_argument("--pixel-two-statistic-direction-power", type=float,
                        default=8.0)
    parser.add_argument("--pixel-two-statistic-slope", type=float, default=8.0)
    parser.add_argument("--pixel-two-statistic-center", type=float, default=0.4)
    parser.add_argument("--pixel-reliability-loss-weight", type=float,
                        default=0.2)
    parser.add_argument("--fusion-lr-multiplier", type=float, default=1.0,
                        help="learning-rate multiplier for global fusion scales")
    parser.add_argument("--validation-scene-fusion", action="store_true",
                        help=("fit one bounded continuous branch weight on "
                              "the validation split at each candidate epoch"))
    parser.add_argument("--validation-fusion-min-alpha", type=float,
                        default=0.10)
    parser.add_argument("--validation-fusion-max-alpha", type=float,
                        default=0.90)
    parser.add_argument("--validation-fusion-shrinkage", type=float,
                        default=1.0,
                        help=("retain this fraction of the validation-fitted "
                              "deviation from equal fusion; 0 is equal fusion "
                              "and 1 is the unregularized fitted weight"))
    parser.add_argument("--validation-reliability-fusion", action="store_true",
                        help=("combine a validation-calibrated scene prior with "
                              "per-sample calibrated confidence evidence"))
    parser.add_argument("--staged-holdout-gate", action="store_true",
                        help=("split the val50 holdout equally into a gate-fit "
                              "set and an untouched model-selection set; fit "
                              "temperatures and a frozen-branch sample gate"))
    parser.add_argument("--structure-fusion",
                        choices=("none", "scene", "scene_competence",
                                 "scene_train_structure", "pixel"),
                        default="none",
                        help=("use centre/context coherence as one scene-level "
                              "weight or as per-pixel staged-gate inputs"))
    parser.add_argument("--reliability-local-strength", type=float,
                        default=2.0)
    parser.add_argument("--pixel-main-loss-weight", type=float, default=1.0)
    parser.add_argument("--pixel-branch-loss-weight", type=float, default=1.0)
    parser.add_argument("--pixel-fused-loss-weight", type=float, default=1.0,
                        help="CE weight of training-time fused prediction")
    parser.add_argument("--disable-embedded-physics-routing", action="store_true",
                        help="train the embedded physical head but report main-head "
                             "predictions without its inference-time routing")
    parser.add_argument("--disable-embedded-physics-branch", action="store_true",
                        help="remove fitting, supervision and routing of the embedded physical branch")
    parser.add_argument("--aux-decay-start", type=int, default=0,
                        help="linearly decay both auxiliary CE weights to zero after this epoch; 0 disables")
    parser.add_argument("--physics-margin", type=float, default=0.0)
    parser.add_argument("--learned-fusion-min-confidence", type=float,
                        default=0.5)
    parser.add_argument("--learned-fusion-conflict-margin", type=float,
                        default=0.15)
    parser.add_argument(
        "--disable-learned-difference-routing", action="store_true",
        help=("remove the learned centre-difference permission decision; "
              "main-vs-physics confidence routing remains active"))
    parser.add_argument("--force-physics", action="store_true",
                        help="enable the train-only calibrator for a pure "
                             "ablation variant without changing its network")
    parser.add_argument("--disable-physics", action="store_true",
                        help="disable the external calibrator for a strict "
                             "network-only control")
    parser.add_argument("--physics-temporal-unreliable", action="store_true",
                        help="let forward/reverse class disagreement mark the "
                             "network prediction unreliable during routing")
    parser.add_argument("--reliability-mode", default="joint",
                        choices=("main_only", "joint", "individual", "scene", "conservative",
                                 "decoupled", "decoupled_dropout",
                                 "decoupled_hard", "decoupled_hard_dropout",
                                 "standardized_dropout", "embedded_logistic",
                                 "embedded_logistic_tune",
                                 "embedded_logistic_temporal",
                                 "embedded_logistic_residual",
                                 "embedded_logistic_feature_residual",
                                 "embedded_logistic_learned_consensus",
                                 "embedded_logistic_learned_adaptive"),
                        help="end-to-end reliability ablation/fusion mode")
    parser.add_argument("--prototype-weight", type=float, default=0.03,
                        help="train-only class geometry weight for compact_signed_prototype")
    parser.add_argument("--swap-consistency-weight", type=float, default=1.0,
                        help="weight of forward/reverse date-order JS consistency")
    parser.add_argument("--single-pass", action="store_true",
                        help=("run only F(date1,date2) in both training and "
                              "evaluation; no reverse-date model call"))
    parser.add_argument("--four-cross-spatial", action="store_true",
                        help=("inside one model call aggregate horizontal and "
                              "vertical VRWKV scans for both T1,T2 and T2,T1 "
                              "interleaving orders; adds no parameters"))
    parser.add_argument("--horizontal-bidirectional-spatial", action="store_true",
                        help=("inside one model call use only horizontal VRWKV "
                              "scans for T1,T2 and T2,T1 orders"))
    parser.add_argument("--horizontal-forward-only-spatial", action="store_true",
                        help=("inside one model call use only the horizontal "
                              "T1,T2 interleaving order"))
    parser.add_argument("--horizontal-vertical-forward-spatial", action="store_true",
                        help=("use T1,T2 horizontal row-major and true vertical "
                              "column-major pixel-interleaved scans"))
    parser.add_argument("--learn-axis-balance", action="store_true",
                        help="learn one bounded horizontal/vertical four-cross mixture")
    parser.add_argument("--vat-weight", type=float, default=0.2,
                        help="weight of change-space virtual adversarial training")
    parser.add_argument("--prototype-decay-start", type=int, default=0,
                        help="if positive, linearly decay batch prototype loss to zero from this epoch")
    parser.add_argument("--prototype-memory-weight", type=float, default=0.0,
                        help="train-only cross-batch labelled prototype weight")
    parser.add_argument("--prototype-memory-momentum", type=float, default=0.9)
    parser.add_argument("--prototype-memory-warmup", type=int, default=0,
                        help="linearly ramp memory loss over this many epochs")
    parser.add_argument("--prototype-memory-delay", type=int, default=0,
                        help="epochs that update memory without applying its loss")
    parser.add_argument("--prototype-codebook-size", type=int, default=1,
                        help="EMA prototypes per class; one reproduces class-centre memory")
    parser.add_argument("--prototype-codebook-temperature", type=float, default=0.10,
                        help="soft assignment temperature for multi-prototype memory")
    parser.add_argument("--prototype-global-bootstrap", action="store_true",
                        help="initialize codebook from one full labelled-train pass after delay")
    parser.add_argument("--supervised-contrastive-weight", type=float, default=0.0,
                        help="class-balanced train-only pairwise representation loss")
    parser.add_argument("--edge-init", type=float, default=-1.0)
    parser.add_argument("--detail-init", type=float, default=-1.0)
    parser.add_argument("--change-residual-init", type=float, default=-1.0)
    parser.add_argument("--disable-sobel-modulation", action="store_true",
                        help="ablate Sobel modulation between Mamba and VRWKV")
    parser.add_argument("--disable-change-residual-gate", action="store_true",
                        help="ablate the post-VRWKV absolute-change residual gate")
    parser.add_argument("--disable-mamba-module", action="store_true",
                        help="ablate all grouped Mamba spectral modelling")
    parser.add_argument("--disable-vrwkv-module", action="store_true",
                        help="ablate VRWKV together with its change residual gate")
    parser.add_argument("--disable-pixel-spectral-grouping", action="store_true",
                        help="retain only global pooling in the centre-spectrum branch")
    parser.add_argument("--spectral-branch-only-ablation", action="store_true",
                        help="execute and supervise only the grouped centre-spectrum branch")
    parser.add_argument("--raw-center-difference-only", action="store_true",
                        help="classify the signed centre difference T2-T1 directly")
    parser.add_argument("--spatial-branch-only-ablation", action="store_true",
                        help="execute and supervise only the spatial VRWKV branch")
    parser.add_argument("--disable-pixel-global-pooling", action="store_true",
                        help="retain band positions instead of global spectral pooling")
    parser.add_argument("--disable-pixel-global-component", action="store_true",
                        help="retain grouped aggregation but remove global aggregation")
    parser.add_argument("--include-pixel-ungrouped-component", action="store_true",
                        help="concatenate per-band relation features with grouped features")
    parser.add_argument("--pixel-spectral-remove-shared-encoder", action="store_true",
                        help="replace the centre spectral encoder with raw spectra")
    parser.add_argument("--pixel-spectral-remove-mean", action="store_true",
                        help="with raw spectra, retain only the absolute difference")
    parser.add_argument("--pixel-spectral-raw-grouping", action="store_true",
                        help="group-pool the raw interaction along the spectral axis")
    parser.add_argument("--pixel-spectral-adaptive-multiscale", action="store_true",
                        help=("replace the centre branch by two-evidence "
                              "sample-adaptive multi-scale spectral modelling"))
    parser.add_argument("--pixel-spectral-dense-kernel9", action="store_true",
                        help=("replace the dilated 3-tap long spectral branch "
                              "with a dense kernel-size-9 convolution"))
    parser.add_argument("--stable-residual-init", type=float, default=None,
                        help=("enable the reliability-weighted shared-state "
                              "residual; omitted preserves the exact base"))
    parser.add_argument("--spatial-scale-aug", type=float, default=0.0,
                        help="probability of synchronized 5/7-to-9 patch scaling")
    parser.add_argument("--hard-example-weight", type=float, default=0.0,
                        help="class-balanced CVaR weight on difficult labelled samples")
    parser.add_argument("--hard-example-fraction", type=float, default=0.5)
    parser.add_argument("--spatial-tta", choices=("none", "transpose"),
                        default="none",
                        help="label-free axis-exchange symmetrization at evaluation")
    parser.add_argument("--same-class-mix-prob", type=float, default=0.0)
    parser.add_argument("--same-class-mix-strength", type=float, default=0.2)
    parser.add_argument("--midpoint-style-prob", type=float, default=0.5,
                        help="probability of paired midpoint style mixing")
    parser.add_argument("--midpoint-style-strength", type=float, default=0.15,
                        help="strength of paired midpoint style mixing")
    parser.add_argument("--snapshot-ensemble", action="store_true",
                        help="fixed online probability average at epochs 41/51/61")
    parser.add_argument("--spectral-group-mask-prob", type=float, default=0.0,
                        help="probability of paired contiguous-band masking during training")
    parser.add_argument("--spectral-group-mask-fraction", type=float, default=0.125,
                        help="fraction of adjacent bands removed by paired masking")
    parser.add_argument("--spectral-gain-prob", type=float, default=0.0,
                        help="probability of paired smooth spectral gain augmentation")
    parser.add_argument("--spectral-gain-strength", type=float, default=0.15,
                        help="maximum smooth multiplicative spectral deviation")
    parser.add_argument("--spectral-offset-prob", type=float, default=0.0,
                        help="probability of paired smooth additive spectral drift")
    parser.add_argument("--spectral-offset-strength", type=float, default=0.05,
                        help="maximum smooth additive drift in standardized units")
    parser.add_argument("--spectral-style-warmup", type=int, default=0,
                        help="epochs used to ramp paired gain/offset strength; zero disables")
    parser.add_argument("--spatial-illumination-prob", type=float, default=0.0,
                        help="probability of paired smooth spatial illumination augmentation")
    parser.add_argument("--spatial-illumination-strength", type=float, default=0.10,
                        help="maximum multiplicative spatial illumination deviation")
    parser.add_argument("--temporal-feature-weight", type=float, default=0.0,
                        help="weight for date-order invariant feature alignment")
    parser.add_argument("--model-ema-decay", type=float, default=0.0,
                        help="parameter EMA decay; zero disables EMA evaluation")
    parser.add_argument("--model-ema-start", type=int, default=1,
                        help="first epoch included in parameter EMA")
    parser.add_argument("--sam-rho", type=float, default=-1.0,
                        help="SAM radius override; negative uses variant default")
    parser.add_argument("--boundary-loss-weight", type=float, default=0.0,
                        help="input-gradient weighted boundary supervision")
    parser.add_argument("--physics-mode", choices=("center", "grouped",
                        "grouped_oof", "grouped_spatial"),
                        default="grouped")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--experiment-group", default=None,
                        help="store this scheme's seeds under logs/results/<group>/")
    parser.add_argument("--init-state", default=None,
                        help="optional untrained initialization state for refactor equivalence audits")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_labelled_pool(positions, labels, protocol, seed):
    """Stratify the fixed labelled pool without touching the test set.

    A separate RNG stream makes this subdivision reproducible while leaving
    the historical dataset split and model RNG stream unchanged.
    """
    if protocol == "test_peak_1pct":
        return positions, labels, None, None
    validation_fraction = {
        "val10_of_1pct": 0.10,
        "val50_of_1pct": 0.50,
    }[protocol]
    rng = np.random.RandomState(seed + 104729)
    train_indices, validation_indices = [], []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        indices = indices[rng.permutation(len(indices))]
        validation_count = max(1, int(round(validation_fraction * len(indices))))
        validation_count = min(validation_count, len(indices) - 1)
        validation_indices.extend(indices[:validation_count])
        train_indices.extend(indices[validation_count:])
    train_indices = np.asarray(train_indices, dtype=np.int64)
    validation_indices = np.asarray(validation_indices, dtype=np.int64)
    return (positions[train_indices], labels[train_indices],
            positions[validation_indices], labels[validation_indices])


def split_gate_and_selection(positions, labels, seed):
    """Split the held-out half class-wise; neither part trains the branches."""
    rng = np.random.RandomState(seed + 130363)
    gate_indices, selection_indices = [], []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        indices = indices[rng.permutation(len(indices))]
        gate_count = max(1, len(indices) // 2)
        gate_count = min(gate_count, len(indices) - 1)
        gate_indices.extend(indices[:gate_count])
        selection_indices.extend(indices[gate_count:])
    gate_indices = np.asarray(gate_indices, dtype=np.int64)
    selection_indices = np.asarray(selection_indices, dtype=np.int64)
    return (positions[gate_indices], labels[gate_indices],
            positions[selection_indices], labels[selection_indices])


def exclude_test_near_labelled(test_positions, test_labels,
                               labelled_positions, radius):
    """Remove test centres whose patch contains a labelled-pool centre."""
    if radius < 0:
        return test_positions, test_labels, np.ones(len(test_labels), bool)
    occupied = {tuple(position) for position in np.asarray(labelled_positions)}
    keep = np.fromiter(
        (not any((row + dr, column + dc) in occupied
                 for dr in range(-radius, radius + 1)
                 for dc in range(-radius, radius + 1))
         for row, column in np.asarray(test_positions)),
        dtype=bool, count=len(test_positions))
    return test_positions[keep], test_labels[keep], keep


def midpoint_style_mix(x1, x2, probability=0.5, strength=0.15):
    if torch.rand((), device=x1.device) >= probability:
        return x1, x2
    midpoint, delta = 0.5 * (x1 + x2), x2 - x1
    mean = midpoint.mean(2, keepdim=True)
    scale = midpoint.std(2, keepdim=True, unbiased=False).clamp_min(1e-4)
    donor = torch.randperm(x1.shape[0], device=x1.device)
    styled = (midpoint - mean) / scale * scale[donor] + mean[donor]
    mixed = midpoint + strength * (styled - midpoint)
    return ((mixed - 0.5 * delta).clamp(0.0, 1.0),
            (mixed + 0.5 * delta).clamp(0.0, 1.0))


def synchronized_scale_view(x1, x2, probability=0.0):
    """Change receptive field without breaking temporal correspondence."""
    if probability <= 0 or torch.rand((), device=x1.device) >= probability:
        return x1, x2
    side = int(round(x1.shape[-1] ** 0.5))
    if side * side != x1.shape[-1] or side < 7:
        return x1, x2
    crop = 5 if bool(torch.rand((), device=x1.device) < 0.5) else 7
    offset = (side - crop) // 2
    def scaled(x):
        image = x.reshape(x.shape[0], x.shape[1], side, side)
        image = image[:, :, offset:offset + crop, offset:offset + crop]
        return F.interpolate(image, size=(side, side), mode="bilinear",
                             align_corners=False).flatten(2)
    return scaled(x1), scaled(x2)


def same_class_pair_mix(x1, x2, target, probability=0.0, strength=0.2):
    """Vicinal paired samples while preserving labels and temporal alignment."""
    if probability <= 0:
        return x1, x2
    donor = torch.arange(len(target), device=target.device)
    for label in target.unique(sorted=True):
        indices = (target == label).nonzero(as_tuple=False).flatten()
        donor[indices] = indices[torch.randperm(len(indices), device=x1.device)]
    selected = (torch.rand(len(target), 1, 1, device=x1.device)
                < probability).to(x1.dtype)
    amount = selected * strength * torch.rand(
        len(target), 1, 1, device=x1.device)
    return (x1 + amount * (x1[donor] - x1),
            x2 + amount * (x2[donor] - x2))


def js_divergence(first, second):
    p = F.softmax(first, 1).clamp_min(1e-7)
    q = F.softmax(second, 1).clamp_min(1e-7)
    mixture = 0.5 * (p + q)
    return 0.5 * ((p * (p.log() - mixture.log())).sum(1).mean()
                  + (q * (q.log() - mixture.log())).sum(1).mean())


def prototype_separation_loss(feature, target, margin=0.15):
    """Compact labelled geometry; uses only classes present in this train batch."""
    feature = F.normalize(feature, dim=1)
    centers, within = [], []
    for label in target.unique(sorted=True):
        members = feature[target == label]
        center = F.normalize(members.mean(0, keepdim=True), dim=1)
        centers.append(center)
        within.append((1.0 - members @ center.t()).mean())
    loss = torch.stack(within).mean()
    if len(centers) > 1:
        centers = torch.cat(centers, dim=0)
        similarity = centers @ centers.t()
        mask = ~torch.eye(len(centers), dtype=torch.bool,
                          device=feature.device)
        loss = loss + F.relu(similarity[mask] - margin).mean()
    return loss


def classwise_hard_loss(logits, target, fraction=0.5):
    """CVaR inside each class, preventing the majority class dominating."""
    per_sample = F.cross_entropy(logits, target, reduction="none",
                                 label_smoothing=0.02)
    terms = []
    for label in target.unique(sorted=True):
        values = per_sample[target == label]
        count = max(1, int(math.ceil(len(values) * fraction)))
        terms.append(values.topk(count, sorted=False).values.mean())
    return torch.stack(terms).mean()


def supervised_contrastive_loss(feature, target, temperature=0.10):
    """Use all labelled same-class relations without collapsing to one centre."""
    feature = F.normalize(feature, dim=1)
    logits = feature @ feature.t() / temperature
    eye = torch.eye(len(target), dtype=torch.bool, device=feature.device)
    positive = target[:, None].eq(target[None, :]) & ~eye
    logits = logits - logits.max(1, keepdim=True).values.detach()
    exp_logits = logits.exp().masked_fill(eye, 0.0)
    log_probability = logits - exp_logits.sum(1, keepdim=True).clamp_min(1e-8).log()
    valid = positive.any(1)
    anchor_loss = -(log_probability * positive).sum(1) / positive.sum(1).clamp_min(1)
    terms = []
    for label in target.unique(sorted=True):
        selected = valid & target.eq(label)
        if bool(selected.any()):
            terms.append(anchor_loss[selected].mean())
    return torch.stack(terms).mean() if terms else feature.sum() * 0.0


def prototype_memory_loss(feature, target, memory):
    """Pull features toward EMA centres formed only by earlier train batches."""
    feature = F.normalize(feature, dim=1)
    terms = []
    for label in target.unique(sorted=True):
        index = int(label)
        initialized = memory["initialized"][index]
        if initialized.ndim == 0 and bool(initialized):
            # Clone because the EMA is updated before backward; autograd must
            # retain the exact historical centre used by this batch's loss.
            center = memory["centers"][index].detach().clone()
            terms.append((1.0 - feature[target == label]
                          @ center).mean())
        elif initialized.ndim > 0 and bool(initialized.any()):
            centers = memory["centers"][index, initialized].detach().clone()
            similarity = feature[target == label] @ centers.t()
            assignment = F.softmax(
                similarity / memory["temperature"], dim=1).detach()
            terms.append((assignment * (1.0 - similarity)).sum(1).mean())
    return (torch.stack(terms).mean() if terms
            else feature.sum() * 0.0)


@torch.no_grad()
def update_prototype_memory(feature, target, memory):
    feature = F.normalize(feature.detach(), dim=1)
    for label in target.unique(sorted=True):
        index = int(label)
        members = feature[target == label]
        initialized = memory["initialized"][index]
        if initialized.ndim == 0:
            center = F.normalize(members.mean(0), dim=0)
            if bool(initialized):
                center = F.normalize(
                    memory["momentum"] * memory["centers"][index]
                    + (1.0 - memory["momentum"]) * center, dim=0)
            memory["centers"][index].copy_(center)
            memory["initialized"][index] = True
            continue
        # Deterministic farthest-first seeding exposes distinct labelled modes
        # without labels beyond the declared 1% training subset.
        if not bool(initialized.all()):
            count = min(len(members), memory["centers"].shape[1])
            chosen = [0]
            for _ in range(1, count):
                similarity = members @ members[chosen].t()
                chosen.append(int(similarity.max(1).values.argmin()))
            memory["centers"][index, :count].copy_(members[chosen])
            memory["initialized"][index, :count] = True
            initialized = memory["initialized"][index]
        centers = memory["centers"][index, initialized]
        assignment = (members @ centers.t()).argmax(1)
        active_indices = initialized.nonzero(as_tuple=False).flatten()
        for local_index, center_index in enumerate(active_indices):
            assigned = members[assignment == local_index]
            if len(assigned) == 0:
                continue
            center = F.normalize(assigned.mean(0), dim=0)
            center = F.normalize(
                memory["momentum"] * memory["centers"][index, center_index]
                + (1.0 - memory["momentum"]) * center, dim=0)
            memory["centers"][index, center_index].copy_(center)


@torch.no_grad()
def bootstrap_prototype_memory(model, loader, memory):
    """Seed centres from every labelled training sample, never test data.

    Batch-local farthest-first initialization is sensitive to which scarce
    changed samples happen to enter the first mini-batch.  A single ordered
    pass over the declared labelled split gives every mode an equal chance to
    seed the codebook while leaving later EMA updates unchanged.
    """
    was_training = model.training
    model.eval()
    features, targets = [], []
    for x1, x2, target in loader:
        x1, x2 = x1.cuda(), x2.cuda()
        forward = model(x1, x2)["feature"]
        reverse = model(x2, x1)["feature"]
        features.append(F.normalize(0.5 * (forward + reverse), dim=1))
        targets.append(target.cuda())
    feature = torch.cat(features)
    target = torch.cat(targets)
    for label in target.unique(sorted=True):
        index = int(label)
        members = feature[target == label]
        initialized = memory["initialized"][index]
        if initialized.ndim == 0:
            memory["centers"][index].copy_(
                F.normalize(members.mean(0), dim=0))
            memory["initialized"][index] = True
            continue
        count = min(len(members), memory["centers"].shape[1])
        # Start from the point least similar to the class mean, then maximize
        # minimum angular distance. This is deterministic and order-neutral.
        class_mean = F.normalize(members.mean(0), dim=0)
        chosen = [int((members @ class_mean).argmin())]
        for _ in range(1, count):
            similarity = members @ members[chosen].t()
            chosen.append(int(similarity.max(1).values.argmin()))
        memory["centers"][index, :count].copy_(members[chosen])
        memory["initialized"][index, :count] = True
    model.train(was_training)


@torch.no_grad()
def refresh_pixel_training_prototypes(model, loader):
    """Compute exact class means from all and only optimizer-training centres."""
    branch = model.pixel_spectral_branch
    if (branch is None or
            not getattr(branch, "use_training_prototypes", False)):
        return None
    was_training = model.training
    model.eval()
    representations, targets = [], []
    for first_patch, second_patch, target in loader:
        first_patch = first_patch.cuda()
        second_patch = second_patch.cuda()
        if first_patch.ndim == 3:
            center = first_patch.shape[2] // 2
            first = first_patch[:, :, center]
            second = second_patch[:, :, center]
        elif first_patch.ndim == 4:
            center_y = first_patch.shape[2] // 2
            center_x = first_patch.shape[3] // 2
            first = first_patch[:, :, center_y, center_x]
            second = second_patch[:, :, center_y, center_x]
        else:
            raise RuntimeError("unexpected patch tensor rank")
        representation, _ = branch(first, second)
        representations.append(representation)
        targets.append(target.cuda())
    representation = torch.cat(representations)
    target = torch.cat(targets)
    for class_index in range(2):
        members = representation[target == class_index]
        if len(members) == 0:
            raise RuntimeError(
                f"training split has no samples for class {class_index}")
        branch.training_prototypes[class_index].copy_(members.mean(0))
    branch.training_prototypes_ready.fill_(True)
    model.train(was_training)
    return {
        "samples": int(len(target)),
        "changed": int((target == 0).sum()),
        "unchanged": int((target == 1).sum()),
        "cosine": float(F.cosine_similarity(
            branch.training_prototypes[0:1],
            branch.training_prototypes[1:2]).item())}


def supervised_objective(model, x1, x2, target, criterion,
                         prototype_weight=0.03, prototype_memory=None,
                         prototype_memory_weight=0.0,
                         update_memory=False, hard_example_weight=0.0,
                         hard_example_fraction=0.5,
                         temporal_feature_weight=0.0,
                         boundary_loss_weight=0.0,
                         supervised_contrastive_weight=0.0,
                         swap_consistency_weight=1.0,
                         coarse_aux_weight=0.3,
                         context_aux_weight=0.10,
                         embedded_physics_aux_weight=0.02,
    learned_difference_aux_weight=0.02):
    forward = model(x1, x2)
    # Strict efficiency ablation: preserve every loss coefficient and output
    # path, but do not execute the network a second time. Reusing `forward`
    # makes all two-order averages algebraically identical to their one-order
    # value and makes the date-order JS exactly zero.
    single_pass = getattr(model, "single_pass", False)
    reverse = forward if single_pass else model(x2, x1)
    # Auxiliary weights denote their total contribution, independently of
    # whether one or two complete detector passes are used.  This removes the
    # historical accidental double-count in single-pass mode while retaining
    # the selected base's effective weights (0.10 context, 0.02 physics).
    order_outputs = (forward,) if single_pass else (forward, reverse)
    if forward["pixel_spectral_logits"] is None:
        prediction_key = "main_logits"
        main_component = 0.5 * (criterion(forward[prediction_key], target)
                                + criterion(reverse[prediction_key], target))
        loss = main_component
        model._last_classification_losses = {
            "fused_loss": None,
            "vrwkv_loss": main_component.detach(),
            "spectral_loss": None,
        }
    else:
        fused_component = 0.5 * (
            criterion(forward["fused_logits"], target)
            + criterion(reverse["fused_logits"], target))
        main_component = 0.5 * (
            criterion(forward["main_logits"], target)
            + criterion(reverse["main_logits"], target))
        spectral_component = 0.5 * (
            criterion(forward["pixel_spectral_logits"], target)
            + criterion(reverse["pixel_spectral_logits"], target))
        loss = model.pixel_fused_loss_weight * fused_component
        loss = loss + model.pixel_main_loss_weight * main_component
        loss = loss + model.pixel_branch_loss_weight * spectral_component
        # Detached bookkeeping only. These are the exact three classification
        # terms used above and therefore do not alter the autograd graph.
        model._last_classification_losses = {
            "fused_loss": fused_component.detach(),
            "vrwkv_loss": main_component.detach(),
            "spectral_loss": spectral_component.detach(),
        }
        if forward["pixel_reliability_weights"] is not None:
            for output in order_outputs:
                main_error = F.cross_entropy(
                    output["main_logits"], target, reduction="none").detach()
                spectral_error = F.cross_entropy(
                    output["pixel_spectral_logits"], target,
                    reduction="none").detach()
                main_target = torch.exp(-main_error).clamp(0.05, 0.95)
                spectral_target = torch.exp(-spectral_error).clamp(0.05, 0.95)
                reliability = output["pixel_reliability_weights"]
                reliability_loss = (
                    F.binary_cross_entropy(reliability[:, 0], main_target)
                    + F.binary_cross_entropy(
                        reliability[:, 1], spectral_target))
                loss = loss + (model.pixel_reliability_loss_weight
                               / len(order_outputs)) * reliability_loss
    for output in order_outputs:
        loss = loss + embedded_physics_aux_weight / len(order_outputs) * criterion(
            output["physics_logits"], target)
        loss = loss + learned_difference_aux_weight / len(order_outputs) * criterion(
            output["learned_logits"], target)
    if forward["coarse"] is not None:
        loss = loss + coarse_aux_weight * 0.5 * (
            criterion(forward["coarse"], target)
            + criterion(reverse["coarse"], target))
    if forward["context"] is not None:
        for output in order_outputs:
            per_sample = F.cross_entropy(
                output["context"], target, reduction="none")
            loss = loss + context_aux_weight / len(order_outputs) * (per_sample * output[
                "context_reliability"]).sum() / output[
                    "context_reliability"].sum()
    if swap_consistency_weight > 0:
        loss = loss + swap_consistency_weight * js_divergence(
            forward["logits"], reverse["logits"])
    if temporal_feature_weight > 0:
        # Change semantics are invariant to exchanging acquisition order.
        # Align normalized representations, while CE retains class separation.
        forward_feature = F.normalize(forward["feature"], dim=1)
        reverse_feature = F.normalize(reverse["feature"], dim=1)
        loss = loss + temporal_feature_weight * (
            1.0 - (forward_feature * reverse_feature).sum(1)).mean()
    if hard_example_weight > 0:
        loss = loss + hard_example_weight * 0.5 * (
            classwise_hard_loss(forward["logits"], target,
                                hard_example_fraction)
            + classwise_hard_loss(reverse["logits"], target,
                                  hard_example_fraction))
    if boundary_loss_weight > 0:
        side = int(round(x1.shape[-1] ** 0.5))
        difference = (x2 - x1).abs().mean(1).reshape(-1, side, side)
        center = side // 2
        gradient = (
            (difference[:, center, min(center + 1, side - 1)]
             - difference[:, center, max(center - 1, 0)]).abs()
            + (difference[:, min(center + 1, side - 1), center]
               - difference[:, max(center - 1, 0), center]).abs())
        gradient = gradient / (gradient.mean().detach() + 1e-6)
        gradient = gradient.detach().clamp(max=3.0)
        boundary = 0.5 * (
            F.cross_entropy(forward["logits"], target, reduction="none")
            + F.cross_entropy(reverse["logits"], target, reduction="none"))
        loss = loss + boundary_loss_weight * (gradient * boundary).mean()
    if prototype_weight > 0:
        # The same 1% labels constrain representation geometry in both date
        # orders; no test samples, pseudo-labels, memory bank, or inference
        # branch are introduced.
        loss = loss + prototype_weight * 0.5 * (
            prototype_separation_loss(forward["feature"], target)
            + prototype_separation_loss(reverse["feature"], target))
    if supervised_contrastive_weight > 0:
        paired = 0.5 * (forward["feature"] + reverse["feature"])
        loss = loss + supervised_contrastive_weight * \
            supervised_contrastive_loss(paired, target)
    if prototype_memory is not None and prototype_memory_weight > 0:
        paired = 0.5 * (forward["feature"] + reverse["feature"])
        loss = loss + prototype_memory_weight * prototype_memory_loss(
            paired, target, prototype_memory)
        if update_memory:
            update_prototype_memory(paired, target, prototype_memory)
    # Match the validated objective: the forward-date logits anchor VAT and
    # training accuracy; the reverse pass regularizes them through CE+JS.
    return loss, forward["logits"]


def unit_per_sample(vector):
    flat = vector.reshape(vector.shape[0], -1)
    return (flat / flat.norm(2, 1, keepdim=True).clamp_min(1e-8)).reshape_as(vector)


def vat_loss(model, x1, x2, clean_logits, epsilon=0.02, xi=1e-3):
    target = F.softmax(clean_logits.detach(), 1)
    direction = unit_per_sample(torch.randn_like(x1)).requires_grad_(True)
    probe = xi * direction
    probe_logits = model((x1 - 0.5 * probe).clamp(0, 1),
                         (x2 + 0.5 * probe).clamp(0, 1))["logits"]
    divergence = F.kl_div(F.log_softmax(probe_logits, 1), target,
                          reduction="batchmean")
    gradient, = torch.autograd.grad(divergence, direction, only_inputs=True)
    adversarial = epsilon * unit_per_sample(gradient.detach())
    logits = model((x1 - 0.5 * adversarial).clamp(0, 1),
                   (x2 + 0.5 * adversarial).clamp(0, 1))["logits"]
    return F.kl_div(F.log_softmax(logits, 1), target, reduction="batchmean")


def sam_perturb(model, rho):
    norms = [parameter.grad.norm(2) for parameter in model.parameters()
             if parameter.grad is not None]
    scale = rho / (torch.stack(norms).norm(2) + 1e-12)
    perturbations = []
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.grad is None:
                continue
            perturbation = parameter.grad * scale
            parameter.add_(perturbation)
            perturbations.append((parameter, perturbation))
    return perturbations


def sam_restore(perturbations):
    with torch.no_grad():
        for parameter, perturbation in perturbations:
            parameter.sub_(perturbation)


def train_epoch(model, loader, optimizer, scheduler, criterion, sam_rho,
                prototype_weight=0.03, prototype_memory=None,
                prototype_memory_weight=0.0, spatial_scale_aug=0.0,
                hard_example_weight=0.0, hard_example_fraction=0.5,
                same_class_mix_prob=0.0, same_class_mix_strength=0.2,
                spectral_group_mask_prob=0.0,
                spectral_group_mask_fraction=0.125,
                spectral_gain_prob=0.0, spectral_gain_strength=0.15,
                spectral_offset_prob=0.0, spectral_offset_strength=0.05,
                spatial_illumination_prob=0.0,
                spatial_illumination_strength=0.10,
                temporal_feature_weight=0.0, boundary_loss_weight=0.0,
                supervised_contrastive_weight=0.0,
                midpoint_style_prob=0.5,
                midpoint_style_strength=0.15,
                swap_consistency_weight=1.0,
                vat_weight=0.2,
                coarse_aux_weight=0.3,
                context_aux_weight=0.05,
                embedded_physics_aux_weight=0.01,
                learned_difference_aux_weight=0.01):
    model.train()
    total_loss = total_correct = total = 0
    classification_totals = {
        "fused_loss": 0.0,
        "vrwkv_loss": 0.0,
        "spectral_loss": 0.0,
    }
    classification_counts = {key: 0 for key in classification_totals}
    for x1, x2, target in loader:
        x1, x2, target = x1.cuda(), x2.cuda(), target.cuda()
        x1, x2 = midpoint_style_mix(
                x1, x2, midpoint_style_prob, midpoint_style_strength)
        x1, x2 = synchronized_scale_view(x1, x2, spatial_scale_aug)
        x1, x2 = same_class_pair_mix(
            x1, x2, target, same_class_mix_prob,
            same_class_mix_strength)
            # The identical contiguous spectral group is hidden in both dates:
            # this regularizes spectral redundancy without fabricating change.
        if (spectral_group_mask_prob > 0
                    and torch.rand((), device=x1.device)
                    < spectral_group_mask_prob):
                bands = x1.shape[1]
                width = max(1, min(bands - 1, int(round(
                    bands * spectral_group_mask_fraction))))
                start = int(torch.randint(
                    0, bands - width + 1, (), device=x1.device))
                x1 = x1.clone()
                x2 = x2.clone()
                x1[:, start:start + width] = 0
                x2[:, start:start + width] = 0
        if (spectral_gain_prob > 0
                    and torch.rand((), device=x1.device) < spectral_gain_prob):
                # Low-frequency multiplicative response shared by both dates.
                # It simulates smooth sensor/illumination drift without changing
                # which material changed between the paired observations.
                anchors = torch.empty(
                    1, 1, 8, device=x1.device).uniform_(
                        -spectral_gain_strength, spectral_gain_strength)
                gain = 1.0 + F.interpolate(
                    anchors, size=x1.shape[1], mode="linear",
                    align_corners=True).reshape(1, x1.shape[1], 1)
                x1 = x1 * gain
                x2 = x2 * gain
        if (spectral_offset_prob > 0
                    and torch.rand((), device=x1.device) < spectral_offset_prob):
                # Smooth additive response shared by both dates. Sharing is
                # essential: it changes radiometric style but not the label.
                anchors = torch.empty(
                    1, 1, 8, device=x1.device).uniform_(
                        -spectral_offset_strength, spectral_offset_strength)
                offset = F.interpolate(
                    anchors, size=x1.shape[1], mode="linear",
                    align_corners=True).reshape(1, x1.shape[1], 1)
                x1 = x1 + offset
                x2 = x2 + offset
        if (spatial_illumination_prob > 0
                    and torch.rand((), device=x1.device)
                    < spatial_illumination_prob):
                # Per-sample smooth illumination field shared by T1/T2.
                # Mean centering separates it from the global spectral gain.
                tokens = x1.shape[-1]
                side = int(round(tokens ** 0.5))
                if side * side == tokens:
                    anchors = torch.empty(
                        x1.shape[0], 1, 3, 3, device=x1.device).uniform_(
                            -spatial_illumination_strength,
                            spatial_illumination_strength)
                    field = F.interpolate(
                        anchors, size=(side, side), mode="bilinear",
                        align_corners=True)
                    field = field - field.mean(dim=(-2, -1), keepdim=True)
                    gain = (1.0 + field).reshape(x1.shape[0], 1, tokens)
                    x1 = x1 * gain
                    x2 = x2 * gain
        optimizer.zero_grad(set_to_none=True)
        loss, logits = supervised_objective(
            model, x1, x2, target, criterion, prototype_weight,
            prototype_memory, prototype_memory_weight, True,
            hard_example_weight, hard_example_fraction,
            temporal_feature_weight, boundary_loss_weight,
            supervised_contrastive_weight, swap_consistency_weight,
            coarse_aux_weight, context_aux_weight,
            embedded_physics_aux_weight, learned_difference_aux_weight)
        classification_snapshot = dict(getattr(
            model, "_last_classification_losses", {}))
        if vat_weight > 0:
            loss = loss + vat_weight * vat_loss(model, x1, x2, logits)
        loss.backward()
        if sam_rho > 0:
            perturbations = sam_perturb(model, sam_rho)
            optimizer.zero_grad(set_to_none=True)
            perturbed, _ = supervised_objective(
                model, x1, x2, target, criterion, prototype_weight,
                prototype_memory, prototype_memory_weight, False,
                hard_example_weight, hard_example_fraction,
                temporal_feature_weight, boundary_loss_weight,
                supervised_contrastive_weight, swap_consistency_weight,
                coarse_aux_weight, context_aux_weight,
                embedded_physics_aux_weight, learned_difference_aux_weight)
            perturbed.backward()
            sam_restore(perturbations)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += float(loss.detach()) * len(target)
        for key, value in classification_snapshot.items():
            if value is not None:
                classification_totals[key] += float(value) * len(target)
                classification_counts[key] += len(target)
        total_correct += int((logits.argmax(1) == target).sum())
        total += len(target)
    scheduler.step()
    result = {"loss": total_loss / total, "oa": total_correct / total}
    for key, value in classification_totals.items():
        count = classification_counts[key]
        result[key] = None if count == 0 else value / count
    return result


def transpose_patch(x):
    side = int(round(x.shape[-1] ** 0.5))
    if side * side != x.shape[-1]:
        raise ValueError("flattened patch area must be square")
    return x.reshape(x.shape[0], x.shape[1], side, side).transpose(
        2, 3).contiguous().flatten(2)


@torch.no_grad()
def predict(model, loader, criterion, retain_spatial=False,
            spatial_tta="none"):
    model.eval()
    probabilities, targets, first_centers, second_centers = [], [], [], []
    main_probabilities, pixel_probabilities, embedded_take_masks = [], [], []
    pixel_fusion_alphas = []
    physical_context_evidences = []
    structure_features = []
    unreliable_masks = []
    embedded_overrides = 0
    total_loss = total = 0
    for x1, x2, target in loader:
        x1, x2, target = x1.cuda(), x2.cuda(), target.cuda()
        center_index = x1.shape[-1] // 2
        temporal = x2 - x1
        center_temporal = temporal[:, :, center_index]
        context_temporal = temporal.mean(2)
        residual = torch.sqrt(
            ((center_temporal - context_temporal) ** 2).mean(1) + 1e-8)
        temporal_scale = (torch.sqrt(
            (center_temporal ** 2).mean(1) + 1e-8)
            + torch.sqrt((context_temporal ** 2).mean(1) + 1e-8))
        residual = residual / temporal_scale.clamp_min(1e-4)
        cosine = F.cosine_similarity(
            center_temporal, context_temporal, dim=1, eps=1e-6)
        contrast = 0.5 * (
            torch.sqrt(((x1[:, :, center_index] - x1.mean(2)) ** 2).mean(1) + 1e-8)
            + torch.sqrt(((x2[:, :, center_index] - x2.mean(2)) ** 2).mean(1) + 1e-8))
        contrast_scale = 0.5 * (
            torch.sqrt((x1[:, :, center_index] ** 2).mean(1) + 1e-8)
            + torch.sqrt((x2[:, :, center_index] ** 2).mean(1) + 1e-8))
        contrast = contrast / contrast_scale.clamp_min(1e-4)
        structure_features.append(torch.stack(
            (residual, cosine, contrast), dim=1).cpu().numpy())
        embedded_probability = None
        forward = model(x1, x2)
        reverse = (forward if getattr(model, "single_pass", False)
                       else model(x2, x1))
        main_logits = 0.5 * (forward["main_logits"]
                             + reverse["main_logits"])
        network_probability = F.softmax(main_logits, 1)[:, 1]
        pixel_logits = forward["pixel_spectral_logits"]
        pixel_probability = (network_probability if pixel_logits is None else
                             F.softmax(pixel_logits, 1)[:, 1])
        fusion_alpha = forward.get("pixel_fusion_alpha")
        if fusion_alpha is not None:
            pixel_fusion_alphas.append(fusion_alpha.cpu().numpy())
        context_evidence = forward.get("physical_context_evidence")
        if context_evidence is not None:
            physical_context_evidences.append(context_evidence.cpu().numpy())
        take = torch.zeros_like(network_probability, dtype=torch.bool)
        if not getattr(model, "disable_embedded_physics_routing", False):
                # Exactly match the validated external order: first average
                # both temporal orders of the network, then route once with
                # the physical probability from the original date order.
                logits = main_logits
                physics_probability = F.softmax(
                    forward["physics_logits"], 1)[:, 1]
                disagreement = ((network_probability >= 0.5)
                                != (physics_probability >= 0.5))
                network_confidence = 2.0 * (network_probability - 0.5).abs()
                take = (disagreement
                        & (2.0 * (physics_probability - 0.5).abs()
                           > network_confidence
                           + model.embedded_reliability_margin))
                if not getattr(
                        model, "disable_learned_difference_routing", False):
                    learned_probability = F.softmax(
                            forward["learned_logits"], 1)[:, 1]
                    learned_confidence = 2.0 * (
                            learned_probability - 0.5).abs()
                    learned_agreement = ((learned_probability >= 0.5)
                                             == (physics_probability >= 0.5))
                    learned_reliable = (learned_confidence >=
                            model.learned_fusion_min_confidence)
                    conflict_safe = (2.0 * (
                                physics_probability - 0.5).abs()
                                > network_confidence
                                + model.learned_fusion_conflict_margin)
                    take = take & ((learned_agreement & learned_reliable)
                                           | conflict_safe)
                embedded_overrides += int(take.sum().item())
                embedded_probability = torch.where(
                    take, physics_probability, network_probability)
                temporal_unreliable = (
                    forward["main_logits"].argmax(1)
                    != reverse["main_logits"].argmax(1))
        else:
                temporal_unreliable = (forward["logits"].argmax(1)
                                       != reverse["logits"].argmax(1))
                logits = 0.5 * (forward["logits"] + reverse["logits"])
        if spatial_tta == "transpose":
                tx1, tx2 = transpose_patch(x1), transpose_patch(x2)
                transpose_forward = model(tx1, tx2)
                transpose_reverse = (
                    transpose_forward if getattr(model, "single_pass", False)
                    else model(tx2, tx1))
                if embedded_probability is not None:
                    # Apply exactly the same train-only physical reliability
                    # rule to the transposed orientation, then fuse the two
                    # final probabilities.  Previously this branch averaged
                    # logits here but `probability` below still returned the
                    # pre-TTA embedded_probability, silently discarding TTA.
                    transpose_main_logits = 0.5 * (
                        transpose_forward["main_logits"]
                        + transpose_reverse["main_logits"])
                    transpose_network_probability = F.softmax(
                        transpose_main_logits, 1)[:, 1]
                    transpose_physics_probability = F.softmax(
                        transpose_forward["physics_logits"], 1)[:, 1]
                    transpose_disagreement = (
                        (transpose_network_probability >= 0.5)
                        != (transpose_physics_probability >= 0.5))
                    transpose_network_confidence = 2.0 * (
                        transpose_network_probability - 0.5).abs()
                    transpose_take = (
                        transpose_disagreement
                        & (2.0 * (transpose_physics_probability - 0.5).abs()
                           > transpose_network_confidence
                           + model.embedded_reliability_margin))
                    embedded_overrides += int(transpose_take.sum().item())
                    transpose_probability = torch.where(
                        transpose_take, transpose_physics_probability,
                        transpose_network_probability)
                    embedded_probability = 0.5 * (
                        embedded_probability + transpose_probability)
                    bounded = embedded_probability.clamp(1e-7, 1.0 - 1e-7)
                    logits = torch.stack((torch.log1p(-bounded),
                                          torch.log(bounded)), dim=1)
                else:
                    transpose_logits = 0.5 * (transpose_forward["logits"]
                                              + transpose_reverse["logits"])
                    logits = 0.5 * (logits + transpose_logits)
                temporal_unreliable |= (
                    transpose_forward["logits"].argmax(1)
                    != transpose_reverse["logits"].argmax(1))
        total_loss += float(criterion(logits, target)) * len(target)
        total += len(target)
        probability = (F.softmax(logits, 1)[:, 1]
                       if embedded_probability is None
                       else embedded_probability)
        probabilities.append(probability.cpu().numpy())
        main_probabilities.append(network_probability.cpu().numpy())
        pixel_probabilities.append(pixel_probability.cpu().numpy())
        embedded_take_masks.append(take.cpu().numpy())
        targets.append(target.cpu().numpy())
        # Center/grouped calibration needs only one spectrum. Avoid retaining
        # 10--20 GB of duplicated full patches on the larger scenes.
        if retain_spatial:
            first_centers.append(x1.cpu().numpy())
            second_centers.append(x2.cpu().numpy())
        else:
            center = x1.shape[-1] // 2
            first_centers.append(x1[:, :, center].cpu().numpy())
            second_centers.append(x2[:, :, center].cpu().numpy())
        unreliable_masks.append(temporal_unreliable.cpu().numpy())
    return (np.concatenate(probabilities), np.concatenate(targets),
            np.concatenate(first_centers), np.concatenate(second_centers),
            total_loss / total, np.concatenate(unreliable_masks),
            embedded_overrides, np.concatenate(main_probabilities),
            np.concatenate(pixel_probabilities),
            np.concatenate(embedded_take_masks),
            (None if not pixel_fusion_alphas else
             np.concatenate(pixel_fusion_alphas)),
            np.concatenate(structure_features),
            (None if not physical_context_evidences else
             np.concatenate(physical_context_evidences)))


def evaluated_metrics(model, loader, criterion, calibrator,
                      spatial_tta="none", include_predictions=False,
                      physics_temporal_unreliable=False,
                      fusion_alpha_override=None,
                      fusion_rule_override=None):
    (probability, target, first, second, loss, unreliable,
     embedded_overrides, main_probability, pixel_probability,
     embedded_take, pixel_fusion_alpha, structure_feature,
     physical_context_evidence) = predict(
        model, loader, criterion,
        retain_spatial=(calibrator is not None
                        and calibrator.mode == "grouped_spatial"),
        spatial_tta=spatial_tta)
    if fusion_alpha_override is not None:
        alpha = float(fusion_alpha_override)
        probability = ((1.0 - alpha) * main_probability
                       + alpha * pixel_probability)
        pixel_fusion_alpha = np.full_like(probability, alpha, dtype=float)
    if (fusion_rule_override is not None
            and fusion_rule_override.get("type") == "scene_structure_fusion"):
        main_calibrated = temperature_scale_probability(
            main_probability, fusion_rule_override["main_temperature"])
        spectral_calibrated = temperature_scale_probability(
            pixel_probability, fusion_rule_override["spectral_temperature"])
        pixel_fusion_alpha = np.full_like(
            main_probability, fusion_rule_override["context_weight"],
            dtype=np.float64)
        probability = (pixel_fusion_alpha * main_calibrated
                       + (1.0 - pixel_fusion_alpha) * spectral_calibrated)
    elif (fusion_rule_override is not None
            and fusion_rule_override.get("type") == "staged_holdout_gate"):
        probability, pixel_fusion_alpha = staged_gate_probability(
            main_probability, pixel_probability, fusion_rule_override,
            structure_feature)
    elif fusion_rule_override is not None:
        main_calibrated = temperature_scale_probability(
            main_probability, fusion_rule_override["main_temperature"])
        spectral_calibrated = temperature_scale_probability(
            pixel_probability, fusion_rule_override["spectral_temperature"])
        main_confidence = 2.0 * np.abs(main_calibrated - 0.5)
        spectral_confidence = 2.0 * np.abs(spectral_calibrated - 0.5)
        scene_alpha = fusion_rule_override["scene_alpha"]
        scene_logit = np.log(scene_alpha / (1.0 - scene_alpha))
        alpha_logit = (scene_logit
                       + fusion_rule_override["local_strength"]
                       * (spectral_confidence - main_confidence))
        pixel_fusion_alpha = 1.0 / (1.0 + np.exp(-alpha_logit))
        pixel_fusion_alpha = np.clip(pixel_fusion_alpha, 0.1, 0.9)
        probability = ((1.0 - pixel_fusion_alpha) * main_calibrated
                       + pixel_fusion_alpha * spectral_calibrated)
    if (getattr(model, "pixel_complementarity_v5_fusion", False)
            and pixel_fusion_alpha is not None
            and fusion_alpha_override is None
            and fusion_rule_override is None):
        # Aggregate unlabeled local physical evidence over the current scene.
        # Validation and test are processed independently, so no training-set
        # weight or label is transferred to the test scene. Four directional
        # regions provide the natural evidence-consensus order.
        local_context = 1.0 - np.asarray(pixel_fusion_alpha, dtype=np.float64)
        mean_context = float(np.clip(local_context.mean(), 1e-6, 1.0 - 1e-6))
        context_odds = mean_context / (1.0 - mean_context)
        scene_context = context_odds ** 4 / (1.0 + context_odds ** 4)
        pixel_fusion_alpha = np.full_like(
            main_probability, 1.0 - scene_context, dtype=np.float64)
        probability = (scene_context * main_probability
                       + (1.0 - scene_context) * pixel_probability)
    if (getattr(model, "pixel_complementarity_v6_fusion", False)
            and physical_context_evidence is not None
            and fusion_alpha_override is None
            and fusion_rule_override is None):
        # Two dates and ten directional relations determine the evidence
        # scale. The two-date odds product supplies the consensus order.
        scene_evidence = float(np.mean(physical_context_evidence))
        scene_odds = 20.0 * scene_evidence
        scene_context = scene_odds ** 2 / (1.0 + scene_odds ** 2)
        pixel_fusion_alpha = np.full_like(
            main_probability, 1.0 - scene_context, dtype=np.float64)
        probability = (scene_context * main_probability
                       + (1.0 - scene_context) * pixel_probability)
    if (getattr(model, "pixel_complementarity_v7_fusion", False)
            and physical_context_evidence is not None
            and pixel_fusion_alpha is not None
            and fusion_alpha_override is None
            and fusion_rule_override is None):
        # Threshold-free contextual support: retain either strong directional
        # consensus or coherent, non-redundant centre/context information.
        local_context = 1.0 - np.asarray(pixel_fusion_alpha, dtype=np.float64)
        mean_context = float(np.clip(local_context.mean(), 1e-6, 1.0 - 1e-6))
        context_odds = mean_context / (1.0 - mean_context)
        consensus_context = context_odds ** 4 / (1.0 + context_odds ** 4)
        group_count = float(model.pixel_spectral_branch.groups)
        direct_odds = group_count * float(np.mean(physical_context_evidence))
        direct_context = direct_odds / (1.0 + direct_odds)
        scene_context = max(consensus_context, direct_context)
        pixel_fusion_alpha = np.full_like(
            main_probability, 1.0 - scene_context, dtype=np.float64)
        probability = (scene_context * main_probability
                       + (1.0 - scene_context) * pixel_probability)
    if (getattr(model, "pixel_complementarity_v8_fusion", False)
            and physical_context_evidence is not None
            and pixel_fusion_alpha is not None
            and fusion_alpha_override is None
            and fusion_rule_override is None):
        local_context = 1.0 - np.asarray(pixel_fusion_alpha, dtype=np.float64)
        mean_context = float(np.clip(local_context.mean(), 1e-6, 1.0 - 1e-6))
        context_odds = mean_context / (1.0 - mean_context)
        consensus_context = context_odds ** 4 / (1.0 + context_odds ** 4)
        relation_count = 10.0
        direct_odds = relation_count * float(np.mean(physical_context_evidence))
        direct_context = direct_odds / (1.0 + direct_odds)
        scene_context = max(consensus_context, direct_context)
        pixel_fusion_alpha = np.full_like(
            main_probability, 1.0 - scene_context, dtype=np.float64)
        probability = (scene_context * main_probability
                       + (1.0 - scene_context) * pixel_probability)
    if (getattr(model, "pixel_complementarity_v9_fusion", False)
            and physical_context_evidence is not None
            and pixel_fusion_alpha is not None
            and fusion_alpha_override is None
            and fusion_rule_override is None):
        local_context = 1.0 - np.asarray(pixel_fusion_alpha, dtype=np.float64)
        mean_context = float(np.clip(local_context.mean(), 1e-6, 1.0 - 1e-6))
        context_odds = mean_context / (1.0 - mean_context)
        consensus_context = context_odds ** 4 / (1.0 + context_odds ** 4)
        bandpass_context = float(np.mean(physical_context_evidence)) ** 3
        scene_context = max(consensus_context, bandpass_context)
        pixel_fusion_alpha = np.full_like(
            main_probability, 1.0 - scene_context, dtype=np.float64)
        probability = (scene_context * main_probability
                       + (1.0 - scene_context) * pixel_probability)
    if (getattr(model, "pixel_complementarity_v10_fusion", False)
            and physical_context_evidence is not None
            and pixel_fusion_alpha is not None
            and fusion_alpha_override is None
            and fusion_rule_override is None):
        local_context = 1.0 - np.asarray(pixel_fusion_alpha, dtype=np.float64)
        mean_context = float(np.clip(local_context.mean(), 1e-6, 1.0 - 1e-6))
        context_odds = mean_context / (1.0 - mean_context)
        consensus_context = context_odds ** 4 / (1.0 + context_odds ** 4)
        evidence = np.asarray(physical_context_evidence, dtype=np.float64)
        robust_range = np.quantile(evidence, 0.9, axis=0) - np.quantile(
            evidence, 0.1, axis=0)
        range_context = float(np.sqrt(np.prod(np.clip(
            robust_range, 0.0, 1.0))))
        scene_context = max(consensus_context, range_context)
        pixel_fusion_alpha = np.full_like(
            main_probability, 1.0 - scene_context, dtype=np.float64)
        probability = (scene_context * main_probability
                       + (1.0 - scene_context) * pixel_probability)
    if (getattr(model, "pixel_complementarity_v11_fusion", False)
            and not getattr(model, "pixel_batch_scene_fusion", False)
            and not getattr(model, "pixel_batch_full_v11_fusion", False)
            and physical_context_evidence is not None
            and pixel_fusion_alpha is not None
            and fusion_alpha_override is None
            and fusion_rule_override is None):
        local_context = 1.0 - np.asarray(pixel_fusion_alpha, dtype=np.float64)
        mean_context = float(np.clip(local_context.mean(), 1e-6, 1.0 - 1e-6))
        context_odds = mean_context / (1.0 - mean_context)
        consensus_context = context_odds ** 4 / (1.0 + context_odds ** 4)
        evidence = np.asarray(physical_context_evidence, dtype=np.float64)
        joint_support, non_redundancy = evidence[:, 0], evidence[:, 1]
        novelty_odds = (float(model.pixel_spectral_branch.groups)
                        * float(np.mean(joint_support * non_redundancy)))
        novelty_context = novelty_odds / (1.0 + novelty_odds)
        robust_range = np.quantile(evidence, 0.9, axis=0) - np.quantile(
            evidence, 0.1, axis=0)
        range_context = float(np.sqrt(np.prod(np.clip(
            robust_range, 0.0, 1.0))))
        scene_context = max(
            consensus_context, novelty_context, range_context)
        pixel_fusion_alpha = np.full_like(
            main_probability, 1.0 - scene_context, dtype=np.float64)
        probability = (scene_context * main_probability
                       + (1.0 - scene_context) * pixel_probability)
    if (getattr(model, "pixel_complementarity_v12_fusion", False)
            and physical_context_evidence is not None
            and pixel_fusion_alpha is not None
            and fusion_alpha_override is None
            and fusion_rule_override is None):
        local_context = 1.0 - np.asarray(pixel_fusion_alpha, dtype=np.float64)
        mean_context = float(np.clip(local_context.mean(), 1e-6, 1.0 - 1e-6))
        context_odds = mean_context / (1.0 - mean_context)
        consensus_context = context_odds ** 4 / (1.0 + context_odds ** 4)
        evidence = np.asarray(physical_context_evidence, dtype=np.float64)
        joint_support, non_redundancy = evidence[:, 0], evidence[:, 1]
        # A dimensionless redundancy correction: coherent novelty matters
        # more when the centre really adds information beyond its context.
        novelty_odds = (float(model.pixel_spectral_branch.groups)
                        * float(np.mean(joint_support * non_redundancy)))
        novelty_odds *= np.sqrt(
            (float(np.mean(non_redundancy)) + 1e-8)
            / (float(np.mean(joint_support)) + 1e-8))
        novelty_context = novelty_odds / (1.0 + novelty_odds)
        robust_range = np.quantile(evidence, 0.9, axis=0) - np.quantile(
            evidence, 0.1, axis=0)
        range_context = float(np.sqrt(np.prod(np.clip(
            robust_range, 0.0, 1.0))))
        scene_context = max(
            consensus_context, novelty_context, range_context)
        pixel_fusion_alpha = np.full_like(
            main_probability, 1.0 - scene_context, dtype=np.float64)
        probability = (scene_context * main_probability
                       + (1.0 - scene_context) * pixel_probability)
    if (getattr(model, "pixel_complementarity_v13_fusion", False)
            and physical_context_evidence is not None
            and pixel_fusion_alpha is not None
            and fusion_alpha_override is None
            and fusion_rule_override is None):
        local_context = 1.0 - np.asarray(pixel_fusion_alpha, dtype=np.float64)
        mean_context = float(np.clip(local_context.mean(), 1e-6, 1.0 - 1e-6))
        context_odds = mean_context / (1.0 - mean_context)
        consensus_context = context_odds ** 4 / (1.0 + context_odds ** 4)
        evidence = np.asarray(physical_context_evidence, dtype=np.float64)
        joint_support, non_redundancy = evidence[:, 0], evidence[:, 1]
        novelty_odds = (float(model.pixel_spectral_branch.groups)
                        * float(np.mean(joint_support * non_redundancy)))
        # Square-root odds temper uncertain scene evidence toward an equal
        # mixture, without a dataset-specific threshold or fitted constant.
        tempered_odds = np.sqrt(novelty_odds)
        novelty_context = tempered_odds / (1.0 + tempered_odds)
        robust_range = np.quantile(evidence, 0.9, axis=0) - np.quantile(
            evidence, 0.1, axis=0)
        range_context = float(np.sqrt(np.prod(np.clip(
            robust_range, 0.0, 1.0))))
        scene_context = max(
            consensus_context, novelty_context, range_context)
        pixel_fusion_alpha = np.full_like(
            main_probability, 1.0 - scene_context, dtype=np.float64)
        probability = (scene_context * main_probability
                       + (1.0 - scene_context) * pixel_probability)
    if (getattr(model, "pixel_complementarity_v14_fusion", False)
            and physical_context_evidence is not None
            and pixel_fusion_alpha is not None
            and fusion_alpha_override is None
            and fusion_rule_override is None):
        local_context = 1.0 - np.asarray(pixel_fusion_alpha, dtype=np.float64)
        mean_context = float(np.clip(local_context.mean(), 1e-6, 1.0 - 1e-6))
        context_odds = mean_context / (1.0 - mean_context)
        consensus_context = context_odds ** 4 / (1.0 + context_odds ** 4)
        evidence = np.asarray(physical_context_evidence, dtype=np.float64)
        joint_support, non_redundancy = evidence[:, 0], evidence[:, 1]
        inter_group_count = max(1, int(model.pixel_spectral_branch.groups) - 1)
        novelty_odds = (inter_group_count
                        * float(np.mean(joint_support * non_redundancy)))
        novelty_context = novelty_odds / (1.0 + novelty_odds)
        robust_range = np.quantile(evidence, 0.9, axis=0) - np.quantile(
            evidence, 0.1, axis=0)
        range_context = float(np.sqrt(np.prod(np.clip(
            robust_range, 0.0, 1.0))))
        scene_context = max(
            consensus_context, novelty_context, range_context)
        pixel_fusion_alpha = np.full_like(
            main_probability, 1.0 - scene_context, dtype=np.float64)
        probability = (scene_context * main_probability
                       + (1.0 - scene_context) * pixel_probability)
    raw = binary_metrics(target, probability >= 0.5)
    reliability_mask = unreliable if physics_temporal_unreliable else None
    if calibrator is None:
        calibrated_probability, overrides = probability, 0
    else:
        calibrated_probability, overrides = calibrator.route(
            probability, first, second, reliability_mask)
    calibrated = binary_metrics(target, calibrated_probability >= 0.5)
    main_uncorrected = binary_metrics(target, main_probability >= 0.5)
    pixel_spectral = binary_metrics(target, pixel_probability >= 0.5)
    main_class = main_probability >= 0.5
    routed_class = probability >= 0.5
    main_right = main_class == target
    routed_right = routed_class == target
    corrected_to_right = int((embedded_take & ~main_right & routed_right).sum())
    corrected_to_wrong = int((embedded_take & main_right & ~routed_right).sum())
    corrected_unchanged = int((embedded_take & (main_right == routed_right)).sum())
    result = {"loss": loss, "raw": raw, "calibrated": calibrated,
              "main_uncorrected": main_uncorrected,
              "pixel_spectral": pixel_spectral,
              "physics_routed": raw,
              "physics_overrides": overrides,
              "embedded_physics_overrides": embedded_overrides,
              "embedded_physics_override_rate": embedded_overrides / len(target),
              "embedded_corrected_to_right": corrected_to_right,
              "embedded_corrected_to_wrong": corrected_to_wrong,
              "embedded_corrected_unchanged": corrected_unchanged,
              "pixel_fusion_alpha_mean": (None if pixel_fusion_alpha is None
                                            else float(pixel_fusion_alpha.mean())),
              "pixel_fusion_alpha_std": (None if pixel_fusion_alpha is None
                                           else float(pixel_fusion_alpha.std()))}
    if ((getattr(model, "pixel_complementarity_fusion", False)
         or getattr(model, "pixel_complementarity_v2_fusion", False)
         or getattr(model, "pixel_complementarity_v3_fusion", False)
         or getattr(model, "pixel_complementarity_v4_fusion", False)
         or getattr(model, "pixel_complementarity_v5_fusion", False)
         or getattr(model, "pixel_complementarity_v6_fusion", False)
         or getattr(model, "pixel_complementarity_v7_fusion", False)
         or getattr(model, "pixel_complementarity_v8_fusion", False)
         or getattr(model, "pixel_complementarity_v9_fusion", False)
         or getattr(model, "pixel_complementarity_v10_fusion", False)
         or getattr(model, "pixel_complementarity_v11_fusion", False)
         or getattr(model, "pixel_complementarity_v12_fusion", False)
         or getattr(model, "pixel_complementarity_v13_fusion", False)
         or getattr(model, "pixel_complementarity_v14_fusion", False)
         or getattr(model, "pixel_two_statistic_fusion", False)
         or getattr(model, "pixel_geomean_fusion", False)
         or getattr(model, "pixel_decomposed_context_fusion", False))
            and pixel_fusion_alpha is not None):
        spectral_weight = np.asarray(pixel_fusion_alpha, dtype=np.float64)
        vrwkv_weight = 1.0 - spectral_weight
        spectral_class = pixel_probability >= 0.5
        spectral_right = spectral_class == target
        result["fusion_weight_semantics"] = (
            "p=f_vrwkv*w_vrwkv+p_spectral*w_spectral")
        result["spectral_weight_mean"] = float(spectral_weight.mean())
        result["spectral_weight_std"] = float(spectral_weight.std())
        result["spectral_weight_quantiles"] = {
            str(q): float(np.quantile(spectral_weight, q))
            for q in (0.05, 0.25, 0.50, 0.75, 0.95)}
        result["vrwkv_weight_mean"] = float(vrwkv_weight.mean())
        result["vrwkv_weight_std"] = float(vrwkv_weight.std())
        result["vrwkv_weight_quantiles"] = {
            str(q): float(np.quantile(vrwkv_weight, q))
            for q in (0.05, 0.25, 0.50, 0.75, 0.95)}
        result["fusion_vs_vrwkv_corrected_to_right"] = int(
            (~main_right & routed_right).sum())
        result["fusion_vs_vrwkv_corrected_to_wrong"] = int(
            (main_right & ~routed_right).sum())
        result["fusion_vs_spectral_corrected_to_right"] = int(
            (~spectral_right & routed_right).sum())
        result["fusion_vs_spectral_corrected_to_wrong"] = int(
            (spectral_right & ~routed_right).sum())
    if getattr(model, "pixel_main_scale_raw", None) is not None:
        result["pixel_main_scale"] = float(
            (F.softplus(model.pixel_main_scale_raw) + 1e-4).detach().cpu())
        result["pixel_spectral_scale"] = float(
            (F.softplus(model.pixel_spectral_scale_raw) + 1e-4).detach().cpu())
    if include_predictions:
        result["calibrated_probability"] = calibrated_probability
        result["target"] = target
        result["main_probability"] = main_probability
        result["pixel_probability"] = pixel_probability
        result["structure_feature"] = structure_feature
    return result


def fit_validation_scene_alpha(validation_eval, lower=0.10, upper=0.90,
                               shrinkage=1.0):
    """Fit one scene-level soft weight without consulting test data."""
    if not 0.0 < lower <= upper < 1.0:
        raise ValueError("validation fusion bounds must lie strictly in (0,1)")
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("validation fusion shrinkage must be in [0,1]")
    target = np.asarray(validation_eval["target"], dtype=np.float64)
    main = np.asarray(validation_eval["main_probability"], dtype=np.float64)
    spectral = np.asarray(
        validation_eval["pixel_probability"], dtype=np.float64)
    sample_weight = np.zeros_like(target, dtype=np.float64)
    for label in (0, 1):
        mask = target == label
        if not mask.any():
            raise ValueError("validation scene fusion requires both classes")
        sample_weight[mask] = 0.5 / mask.sum()
    best = None
    for alpha in np.linspace(lower, upper, 161):
        probability = np.clip(
            (1.0 - alpha) * main + alpha * spectral, 1e-7, 1.0 - 1e-7)
        log_loss = -np.sum(sample_weight * (
            target * np.log(probability)
            + (1.0 - target) * np.log(1.0 - probability)))
        objective = log_loss + 0.002 * (alpha - 0.5) ** 2
        key = (objective, abs(alpha - 0.5))
        if best is None or key < best[0]:
            best = (key, float(alpha))
    return 0.5 + shrinkage * (best[1] - 0.5)


def temperature_scale_probability(probability, temperature):
    probability = np.clip(np.asarray(probability, dtype=np.float64),
                          1e-7, 1.0 - 1e-7)
    logits = np.log(probability / (1.0 - probability)) / float(temperature)
    return 1.0 / (1.0 + np.exp(-logits))


def balanced_log_loss(target, probability):
    target = np.asarray(target, dtype=np.float64)
    probability = np.clip(np.asarray(probability, dtype=np.float64),
                          1e-7, 1.0 - 1e-7)
    weights = np.zeros_like(target)
    for label in (0, 1):
        mask = target == label
        if not mask.any():
            raise ValueError("reliability fusion requires both classes")
        weights[mask] = 0.5 / mask.sum()
    return float(-np.sum(weights * (
        target * np.log(probability)
        + (1.0 - target) * np.log(1.0 - probability))))


def gate_features(main_probability, spectral_probability,
                  structure_feature=None):
    main = np.clip(np.asarray(main_probability, dtype=np.float64),
                   1e-7, 1.0 - 1e-7)
    spectral = np.clip(np.asarray(spectral_probability, dtype=np.float64),
                       1e-7, 1.0 - 1e-7)
    main_entropy = -(main * np.log(main) + (1.0 - main) * np.log1p(-main))
    spectral_entropy = -(spectral * np.log(spectral)
                         + (1.0 - spectral) * np.log1p(-spectral))
    probability_feature = np.stack((
        main, spectral, main_entropy, spectral_entropy,
        np.abs(main - spectral)), axis=1)
    if structure_feature is None:
        return probability_feature
    structure = np.asarray(structure_feature, dtype=np.float64)
    return np.concatenate((probability_feature, structure), axis=1)


def fit_staged_holdout_gate(gate_eval, seed, hidden=8,
                            use_structure=False):
    """Fit calibration and a tiny gate using only branch-unseen labels."""
    target = np.asarray(gate_eval["target"], dtype=np.float32)
    raw_main = np.asarray(gate_eval["main_probability"])
    raw_spectral = np.asarray(gate_eval["pixel_probability"])
    candidates = np.exp(np.linspace(np.log(0.5), np.log(3.0), 51))
    main_temperature = float(min(
        candidates, key=lambda value: balanced_log_loss(
            target, temperature_scale_probability(raw_main, value))))
    spectral_temperature = float(min(
        candidates, key=lambda value: balanced_log_loss(
            target, temperature_scale_probability(raw_spectral, value))))
    main = temperature_scale_probability(raw_main, main_temperature)
    spectral = temperature_scale_probability(
        raw_spectral, spectral_temperature)
    feature = gate_features(
        main, spectral,
        gate_eval.get("structure_feature") if use_structure else None
    ).astype(np.float32)
    feature_mean = feature.mean(0)
    feature_scale = feature.std(0).clip(1e-4)
    normalized = (feature - feature_mean) / feature_scale

    # CPU training makes the post-hoc stage deterministic and keeps all
    # branch tensors detached. The final layer starts at zero (alpha=0.5).
    generator_state = torch.random.get_rng_state()
    torch.manual_seed(seed + 169087)
    gate = torch.nn.Sequential(torch.nn.Linear(feature.shape[1], hidden),
                               torch.nn.Tanh(),
                               torch.nn.Linear(hidden, 1))
    torch.nn.init.zeros_(gate[-1].weight)
    torch.nn.init.zeros_(gate[-1].bias)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=0.02,
                                  weight_decay=0.02)
    x = torch.from_numpy(normalized)
    y = torch.from_numpy(target)
    main_tensor = torch.from_numpy(main.astype(np.float32))
    spectral_tensor = torch.from_numpy(spectral.astype(np.float32))
    class_weight = torch.where(
        y > 0.5, 0.5 / y.sum().clamp_min(1),
        0.5 / (1.0 - y).sum().clamp_min(1))
    for _ in range(300):
        alpha = torch.sigmoid(gate(x).squeeze(1))
        fused = (alpha * main_tensor
                 + (1.0 - alpha) * spectral_tensor).clamp(1e-6, 1 - 1e-6)
        loss = -(class_weight * (
            y * fused.log() + (1.0 - y) * torch.log1p(-fused))).sum()
        # Mild shrinkage prevents the scarce gate set producing hard routing.
        loss = loss + 0.002 * ((alpha - 0.5) ** 2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    torch.random.set_rng_state(generator_state)
    state = gate.state_dict()
    return {
        "type": "staged_holdout_gate", "main_temperature": main_temperature,
        "uses_structure": bool(use_structure),
        "spectral_temperature": spectral_temperature,
        "feature_mean": feature_mean.tolist(),
        "feature_scale": feature_scale.tolist(),
        "hidden_weight": state["0.weight"].numpy().tolist(),
        "hidden_bias": state["0.bias"].numpy().tolist(),
        "output_weight": state["2.weight"].numpy().tolist(),
        "output_bias": state["2.bias"].numpy().tolist()}


def staged_gate_probability(main_probability, spectral_probability, rule,
                            structure_feature=None):
    main = temperature_scale_probability(
        main_probability, rule["main_temperature"])
    spectral = temperature_scale_probability(
        spectral_probability, rule["spectral_temperature"])
    feature = gate_features(
        main, spectral,
        structure_feature if rule.get("uses_structure", False) else None)
    feature = ((feature - np.asarray(rule["feature_mean"]))
               / np.asarray(rule["feature_scale"]))
    hidden = np.tanh(feature @ np.asarray(rule["hidden_weight"]).T
                     + np.asarray(rule["hidden_bias"]))
    logit = hidden @ np.asarray(rule["output_weight"]).T
    logit = logit[:, 0] + float(rule["output_bias"][0])
    alpha = 1.0 / (1.0 + np.exp(-np.clip(logit, -30.0, 30.0)))
    return alpha * main + (1.0 - alpha) * spectral, alpha


def apply_staged_gate_metrics(evaluation, rule):
    target = np.asarray(evaluation["target"])
    probability, alpha = staged_gate_probability(
        evaluation["main_probability"], evaluation["pixel_probability"], rule,
        evaluation.get("structure_feature"))
    metrics = binary_metrics(target, probability >= 0.5)
    evaluation["loss"] = balanced_log_loss(target, probability)
    evaluation["raw"] = metrics
    evaluation["calibrated"] = metrics
    evaluation["physics_routed"] = metrics
    evaluation["pixel_fusion_alpha_mean"] = float(alpha.mean())
    evaluation["pixel_fusion_alpha_std"] = float(alpha.std())
    evaluation["staged_holdout_gate"] = rule
    for key in ("calibrated_probability", "target", "main_probability",
                "pixel_probability", "structure_feature"):
        evaluation.pop(key, None)
    return evaluation


def fit_scene_structure_rule(gate_eval, train_eval):
    """Create one context weight from every branch-training patch."""
    target = np.asarray(gate_eval["target"])
    raw_main = np.asarray(gate_eval["main_probability"])
    raw_spectral = np.asarray(gate_eval["pixel_probability"])
    candidates = np.exp(np.linspace(np.log(0.5), np.log(3.0), 51))
    main_temperature = float(min(
        candidates, key=lambda value: balanced_log_loss(
            target, temperature_scale_probability(raw_main, value))))
    spectral_temperature = float(min(
        candidates, key=lambda value: balanced_log_loss(
            target, temperature_scale_probability(raw_spectral, value))))
    structure = np.asarray(train_eval["structure_feature"], dtype=np.float64)
    residual, cosine, contrast = np.median(structure, axis=0)
    coherence = 0.5 * (cosine + 1.0) * np.exp(-residual - 0.5 * contrast)
    context_weight = 1.0 / (1.0 + np.exp(-8.0 * (coherence - 0.55)))
    context_weight = float(np.clip(context_weight, 0.10, 0.90))
    return {"type": "scene_structure_fusion",
            "main_temperature": main_temperature,
            "spectral_temperature": spectral_temperature,
            "context_weight": context_weight,
            "median_residual": float(residual),
            "median_cosine": float(cosine),
            "median_contrast": float(contrast),
            "coherence": float(coherence)}


def fit_scene_competence_rule(gate_eval, train_eval):
    """Scene fusion with redundancy neutralisation and holdout competence."""
    target = np.asarray(gate_eval["target"])
    raw_main = np.asarray(gate_eval["main_probability"])
    raw_spectral = np.asarray(gate_eval["pixel_probability"])
    candidates = np.exp(np.linspace(np.log(0.5), np.log(3.0), 51))
    main_temperature = float(min(
        candidates, key=lambda value: balanced_log_loss(
            target, temperature_scale_probability(raw_main, value))))
    spectral_temperature = float(min(
        candidates, key=lambda value: balanced_log_loss(
            target, temperature_scale_probability(raw_spectral, value))))
    main_probability = temperature_scale_probability(raw_main, main_temperature)
    spectral_probability = temperature_scale_probability(
        raw_spectral, spectral_temperature)
    main_aa = float(binary_metrics(target, main_probability >= 0.5)["aa"])
    spectral_aa = float(binary_metrics(
        target, spectral_probability >= 0.5)["aa"])

    structure = np.asarray(train_eval["structure_feature"], dtype=np.float64)
    residual, cosine, contrast = np.median(structure, axis=0)
    coherence = 0.5 * (cosine + 1.0) * np.exp(-residual - 0.5 * contrast)
    old_weight = 1.0 / (1.0 + np.exp(-8.0 * (coherence - 0.55)))

    # Extreme centre/neighbour agreement may indicate redundant context rather
    # than useful extra evidence. Neutralise that case toward equal weighting.
    novelty = float(np.clip(
        1.0 - np.exp(-4.0 * (residual + contrast)), 0.0, 1.0))
    structural_weight = 0.5 + (old_weight - 0.5) * novelty ** 2

    # Correct the prior using branch-unseen labels only. Shrink the correction
    # when the holdout is small and keep the final decision softly bounded.
    holdout_shrinkage = float(len(target) / (len(target) + 200.0))
    competence_delta = holdout_shrinkage * (main_aa - spectral_aa)
    structural_logit = np.log(
        np.clip(structural_weight, 1e-5, 1.0 - 1e-5)
        / np.clip(1.0 - structural_weight, 1e-5, 1.0))
    context_weight = 1.0 / (1.0 + np.exp(-(
        structural_logit + 20.0 * competence_delta)))
    context_weight = float(np.clip(context_weight, 0.10, 0.90))
    return {"type": "scene_structure_fusion",
            "formula": "redundancy_neutralized_holdout_competence_v1",
            "main_temperature": main_temperature,
            "spectral_temperature": spectral_temperature,
            "context_weight": context_weight,
            "median_residual": float(residual),
            "median_cosine": float(cosine),
            "median_contrast": float(contrast),
            "coherence": float(coherence),
            "novelty": novelty,
            "structural_weight": float(structural_weight),
            "holdout_size": int(len(target)),
            "holdout_shrinkage": holdout_shrinkage,
            "holdout_main_aa": main_aa,
            "holdout_spectral_aa": spectral_aa,
            "competence_delta": float(competence_delta)}


def fit_train_structure_rule(train_eval):
    """Fit one scene weight from training-patch structure without holdout labels."""
    structure = np.asarray(train_eval["structure_feature"], dtype=np.float64)
    residual, cosine, contrast = np.median(structure, axis=0)
    coherence = 0.5 * (cosine + 1.0) * np.exp(-residual - 0.5 * contrast)
    old_weight = 1.0 / (1.0 + np.exp(-8.0 * (coherence - 0.55)))
    novelty = float(np.clip(
        1.0 - np.exp(-4.0 * (residual + contrast)), 0.0, 1.0))

    # Context is useful only when it is both reliable and complementary.
    # A very large centre-neighbour residual or a low directional cosine
    # indicates boundary/mixture conflict.  Conversely, an extremely small
    # residual together with a cosine near one means that the neighbourhood
    # mostly repeats the centre spectrum and contributes little new evidence.
    # Four smooth one-sided gates therefore define a broad, interpretable
    # "reliable but non-redundant" interval without consulting any labels.
    sigmoid = lambda value: 1.0 / (1.0 + np.exp(-value))
    direction_reliable = sigmoid((cosine - 0.85) / 0.025)
    residual_reliable = sigmoid((0.30 - residual) / 0.025)
    residual_novel = sigmoid((residual - 0.135) / 0.018)
    direction_novel = sigmoid(((1.0 - cosine) - 0.025) / 0.012)
    complementarity = float(direction_reliable * residual_reliable
                            * residual_novel * direction_novel)

    # Even when context is redundant or locally inconsistent, retain a 35%
    # VRWKV contribution.  Strong complementary evidence can raise it to 85%.
    context_weight = float(np.clip(0.35 + 0.50 * complementarity,
                                   0.35, 0.85))
    return {"type": "scene_structure_fusion",
            "formula": "train_patch_complementarity_band_v2",
            "main_temperature": 1.0,
            "spectral_temperature": 1.0,
            "context_weight": context_weight,
            "median_residual": float(residual),
            "median_cosine": float(cosine),
            "median_contrast": float(contrast),
            "coherence": float(coherence),
            "novelty": novelty,
            "old_context_weight": float(old_weight),
            "direction_reliable": float(direction_reliable),
            "residual_reliable": float(residual_reliable),
            "residual_novel": float(residual_novel),
            "direction_novel": float(direction_novel),
            "complementarity": complementarity,
            "uses_holdout_labels": False,
            "uses_validation_labels": False}


def apply_scene_structure_metrics(evaluation, rule):
    main = temperature_scale_probability(
        evaluation["main_probability"], rule["main_temperature"])
    spectral = temperature_scale_probability(
        evaluation["pixel_probability"], rule["spectral_temperature"])
    probability = (rule["context_weight"] * main
                   + (1.0 - rule["context_weight"]) * spectral)
    metrics = binary_metrics(np.asarray(evaluation["target"]),
                             probability >= 0.5)
    evaluation["loss"] = balanced_log_loss(evaluation["target"], probability)
    evaluation["raw"] = metrics
    evaluation["calibrated"] = metrics
    evaluation["physics_routed"] = metrics
    evaluation["pixel_fusion_alpha_mean"] = rule["context_weight"]
    evaluation["pixel_fusion_alpha_std"] = 0.0
    evaluation["scene_structure_rule"] = rule
    for key in ("calibrated_probability", "target", "main_probability",
                "pixel_probability", "structure_feature"):
        evaluation.pop(key, None)
    return evaluation


def fit_validation_reliability_rule(validation_eval, local_strength=2.0):
    """Fit calibration and a scene prior; retain sample-wise adaptation."""
    target = np.asarray(validation_eval["target"])
    main = np.asarray(validation_eval["main_probability"])
    spectral = np.asarray(validation_eval["pixel_probability"])

    def fit_temperature(probability):
        candidates = np.exp(np.linspace(np.log(0.5), np.log(3.0), 51))
        return float(min(candidates, key=lambda temperature:
            balanced_log_loss(
                target,
                temperature_scale_probability(probability, temperature))))

    main_temperature = fit_temperature(main)
    spectral_temperature = fit_temperature(spectral)
    main_calibrated = temperature_scale_probability(main, main_temperature)
    spectral_calibrated = temperature_scale_probability(
        spectral, spectral_temperature)
    main_loss = balanced_log_loss(target, main_calibrated)
    spectral_loss = balanced_log_loss(target, spectral_calibrated)
    # A smooth scene preference: a 0.10 balanced-NLL advantage corresponds
    # to roughly 73% prior weight, before local sample evidence is applied.
    scene_alpha = 1.0 / (1.0 + np.exp(-(main_loss - spectral_loss) / 0.10))
    scene_alpha = float(np.clip(scene_alpha, 0.20, 0.80))
    return {"main_temperature": main_temperature,
            "spectral_temperature": spectral_temperature,
            "main_validation_loss": main_loss,
            "spectral_validation_loss": spectral_loss,
            "scene_alpha": scene_alpha,
            "local_strength": float(local_strength)}


def apply_reliability_fusion_metrics(validation_eval, rule):
    target = np.asarray(validation_eval["target"])
    main = temperature_scale_probability(
        validation_eval["main_probability"], rule["main_temperature"])
    spectral = temperature_scale_probability(
        validation_eval["pixel_probability"], rule["spectral_temperature"])
    main_confidence = 2.0 * np.abs(main - 0.5)
    spectral_confidence = 2.0 * np.abs(spectral - 0.5)
    scene_logit = np.log(rule["scene_alpha"] / (1.0 - rule["scene_alpha"]))
    alpha = 1.0 / (1.0 + np.exp(-(
        scene_logit + rule["local_strength"]
        * (spectral_confidence - main_confidence))))
    alpha = np.clip(alpha, 0.1, 0.9)
    probability = (1.0 - alpha) * main + alpha * spectral
    metrics = binary_metrics(target, probability >= 0.5)
    validation_eval["loss"] = balanced_log_loss(target, probability)
    validation_eval["raw"] = metrics
    validation_eval["calibrated"] = metrics
    validation_eval["physics_routed"] = metrics
    validation_eval["pixel_fusion_alpha_mean"] = float(alpha.mean())
    validation_eval["pixel_fusion_alpha_std"] = float(alpha.std())
    validation_eval["validation_reliability_rule"] = rule
    for key in ("calibrated_probability", "target", "main_probability",
                "pixel_probability"):
        validation_eval.pop(key, None)
    return validation_eval


def apply_scene_fusion_metrics(validation_eval, alpha):
    target = np.asarray(validation_eval["target"])
    main = np.asarray(validation_eval["main_probability"])
    spectral = np.asarray(validation_eval["pixel_probability"])
    probability = (1.0 - alpha) * main + alpha * spectral
    clipped = np.clip(probability, 1e-7, 1.0 - 1e-7)
    validation_eval["loss"] = float(-np.mean(
        target * np.log(clipped) + (1.0 - target) * np.log(1.0 - clipped)))
    metrics = binary_metrics(target, probability >= 0.5)
    validation_eval["raw"] = metrics
    validation_eval["calibrated"] = metrics
    validation_eval["physics_routed"] = metrics
    validation_eval["pixel_fusion_alpha_mean"] = float(alpha)
    validation_eval["pixel_fusion_alpha_std"] = 0.0
    validation_eval["validation_scene_fusion_alpha"] = float(alpha)
    for key in ("calibrated_probability", "target", "main_probability",
                "pixel_probability"):
        validation_eval.pop(key, None)
    return validation_eval


def main():
    args = arguments()
    if sum((args.pixel_adaptive_fusion, args.pixel_independent_fusion,
            args.pixel_dual_reliability_fusion,
            args.pixel_complementarity_fusion,
            args.pixel_complementarity_v2_fusion,
            args.pixel_complementarity_v3_fusion,
            args.pixel_complementarity_v4_fusion,
            args.pixel_complementarity_v5_fusion,
            args.pixel_complementarity_v6_fusion,
            args.pixel_complementarity_v7_fusion,
            args.pixel_complementarity_v8_fusion,
            args.pixel_complementarity_v9_fusion,
            args.pixel_complementarity_v10_fusion,
            args.pixel_complementarity_v11_fusion,
            args.pixel_complementarity_v12_fusion,
            args.pixel_complementarity_v13_fusion,
            args.pixel_complementarity_v14_fusion,
            args.pixel_two_statistic_fusion,
            args.pixel_geomean_fusion,
            args.pixel_decomposed_context_fusion)) > 1:
        raise ValueError("choose one trainable pixel fusion mode")
    if args.validation_scene_fusion and args.validation_reliability_fusion:
        raise ValueError("choose only one validation fusion strategy")
    if args.staged_holdout_gate and (
            args.validation_scene_fusion or args.validation_reliability_fusion):
        raise ValueError("staged holdout gate is a separate fusion protocol")
    if args.staged_holdout_gate and args.selection_protocol != "val50_of_1pct":
        raise ValueError("staged holdout gate requires val50_of_1pct")
    if (args.structure_fusion not in ("none", "scene_train_structure")
            and not args.staged_holdout_gate):
        raise ValueError("structure fusion requires --staged-holdout-gate")
    if args.structure_fusion == "scene_train_structure" and args.staged_holdout_gate:
        raise ValueError("training-structure fusion must not create a gate holdout")
    if args.validation_scene_fusion or args.validation_reliability_fusion:
        if args.selection_protocol == "test_peak_1pct":
            raise ValueError(
                "validation fusion requires a validation protocol")
        if args.pixel_spectral_groups <= 0:
            raise ValueError(
                "validation fusion requires the learned spectral branch")
        if args.pixel_adaptive_fusion:
            raise ValueError(
                "use fixed training fusion with validation fusion; "
                "do not combine it with the sample-adaptive gate")
    set_seed(args.seed)
    root = Path(__file__).resolve().parent
    name = args.run_name or f"{args.variant}_seed{args.seed}"
    log_root = Path(os.environ.get("VRWKV_LOG_ROOT", root / "logs"))
    result_root = Path(os.environ.get("VRWKV_RESULT_ROOT", root / "results"))
    if args.experiment_group:
        log_root = log_root / args.experiment_group
        result_root = result_root / args.experiment_group
    log_root.mkdir(parents=True, exist_ok=True)
    result_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / f"{name}.log"
    result_path = result_root / f"{name}.json"
    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError("train-fraction must be between zero and one")
    fraction_key = f"{100 * args.train_fraction:g}pct".replace(".", "p")
    split_root = Path(os.environ.get(
        "VRWKV_SPLIT_ROOT", root / "results" / "splits"))
    split_path = (split_root / args.dataset
                  / fraction_key / f"seed{args.seed}.npz")
    split = load_benchmark(args.dataset, args.seed,
                           fraction=args.train_fraction,
                           split_output=split_path,
                           normalization=args.input_normalization,
                           normalization_patch=args.patch,
                           normalization_protocol=args.selection_protocol)
    (fit_positions, fit_labels,
     validation_positions, validation_labels) = split_labelled_pool(
        split.train_positions, split.train_labels,
        args.selection_protocol, args.seed)
    (evaluation_test_positions, evaluation_test_labels,
     evaluation_test_keep) = exclude_test_near_labelled(
        split.test_positions, split.test_labels, split.train_positions,
        args.test_exclusion_radius)
    if len(evaluation_test_labels) == 0:
        raise RuntimeError("test exclusion removed every test sample")
    gate_positions = gate_labels = None
    if args.staged_holdout_gate:
        (gate_positions, gate_labels,
         validation_positions, validation_labels) = split_labelled_pool(
            validation_positions, validation_labels,
            "val50_of_1pct", args.seed + 32452843)
    if args.input_normalization == "train_patches":
        normalization_audit = np.load(split_path, allow_pickle=False)
        normalization_positions = normalization_audit[
            "normalization_fit_positions"]
        if not np.array_equal(normalization_positions, fit_positions):
            raise RuntimeError(
                "normalization fit positions include non-training samples")
    selection_base = Path(os.environ.get(
        "VRWKV_SELECTION_SPLIT_ROOT", root / "results" / "selection_splits"))
    selection_root = (selection_base
                      / args.dataset / fraction_key / args.selection_protocol)
    if args.test_exclusion_radius >= 0:
        selection_root = (selection_root /
                          f"strict_disjoint_r{args.test_exclusion_radius}")
    selection_root.mkdir(parents=True, exist_ok=True)
    selection_path = selection_root / f"seed{args.seed}.npz"
    np.savez_compressed(
        selection_path, dataset=args.dataset, seed=np.int64(args.seed),
        protocol=args.selection_protocol,
        fit_positions=fit_positions, fit_labels=fit_labels,
        validation_positions=(np.empty((0, 2), dtype=np.int64)
                              if validation_positions is None
                              else validation_positions),
        validation_labels=(np.empty((0,), dtype=np.int64)
                           if validation_labels is None
                           else validation_labels),
        gate_positions=(np.empty((0, 2), dtype=np.int64)
                        if gate_positions is None else gate_positions),
        gate_labels=(np.empty((0,), dtype=np.int64)
                     if gate_labels is None else gate_labels),
        test_positions=evaluation_test_positions,
        test_labels=evaluation_test_labels,
        original_test_positions=split.test_positions,
        original_test_labels=split.test_labels,
        test_keep_mask=evaluation_test_keep,
        test_exclusion_radius=np.int64(args.test_exclusion_radius))
    padded_first = reflect_pad(split.first, args.patch)
    padded_second = reflect_pad(split.second, args.patch)
    train_data = LazyPairDataset(padded_first, padded_second,
                                 fit_positions, fit_labels,
                                 args.patch, already_padded=True)
    validation_data = (None if validation_positions is None else
        LazyPairDataset(padded_first, padded_second,
                        validation_positions, validation_labels,
                        args.patch, already_padded=True))
    gate_data = (None if gate_positions is None else
        LazyPairDataset(padded_first, padded_second,
                        gate_positions, gate_labels,
                        args.patch, already_padded=True))
    test_data = LazyPairDataset(padded_first, padded_second,
                                evaluation_test_positions,
                                evaluation_test_labels,
                                args.patch, already_padded=True)
    loose_test_data = LazyPairDataset(
        padded_first, padded_second, split.test_positions, split.test_labels,
        args.patch, already_padded=True)
    train_loader = DataLoader(train_data, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers)
    train_eval_loader = DataLoader(train_data, batch_size=args.test_batch_size,
                                   shuffle=False, num_workers=args.workers)
    validation_loader = (None if validation_data is None else DataLoader(
        validation_data, batch_size=args.test_batch_size,
        shuffle=False, num_workers=args.workers))
    gate_loader = (None if gate_data is None else DataLoader(
        gate_data, batch_size=args.test_batch_size,
        shuffle=False, num_workers=args.workers))
    test_loader = DataLoader(test_data, batch_size=args.test_batch_size,
                             shuffle=False, num_workers=args.workers)
    loose_test_loader = DataLoader(
        loose_test_data, batch_size=args.test_batch_size,
        shuffle=False, num_workers=args.workers)
    # The final architecture embeds its train-only reliability model; no
    # separate post-hoc calibrator is used at test time.
    calibrator = None
    train_first = np.stack([train_data[index][0].numpy()
                            for index in range(len(train_data))])
    train_second = np.stack([train_data[index][1].numpy()
                             for index in range(len(train_data))])
    if args.disable_embedded_physics_branch:
        embedded_calibrator = None
        audit = {"integration": "disabled", "embedded_physics_branch": False}
    else:
        embedded_mode = ("grouped_oof" if args.physics_mode == "grouped_oof"
                         else "grouped")
        embedded_calibrator = TrainOnlyPhysicsCalibrator(0.0, embedded_mode)
        audit = embedded_calibrator.fit(train_first, train_second, fit_labels)
        audit["integration"] = args.reliability_mode
    # Alternating signed decay supplies near/far memory within one WKV call.
    if not (0.0 < args.rwkv_decay_min <= args.rwkv_decay_max):
        raise ValueError("RWKV decay range must satisfy 0 < min <= max")
    os.environ["RWKV_DECAY_INIT"] = "signed_symmetric"
    os.environ["RWKV_DECAY_MIN"] = str(args.rwkv_decay_min)
    os.environ["RWKV_DECAY_MAX"] = str(args.rwkv_decay_max)
    os.environ["RWKV_EDGE_GATE_MODE"] = args.vrwkv_edge_mode
    model = CleanVRWKVCD(
        split.first.shape[2], args.patch, args.local_patch, args.variant,
        reliability_mode=args.reliability_mode,
        spectral_spatial_order=args.spectral_spatial_order,
        spectral_groups=args.spectral_groups,
        mamba_state=args.mamba_state,
        mamba_intra_state=(args.mamba_intra_state or None),
        mamba_inter_state=(args.mamba_inter_state or None),
        mamba_expand=args.mamba_expand,
        mamba_depth=args.mamba_depth,
        vrwkv_depth=args.vrwkv_depth,
        classifier_dropout=args.classifier_dropout,
        spatial_input_dim=args.spatial_input_dim,
        spatial_feature_dim=args.spatial_feature_dim,
        spatial_unshared_input_projection=
        args.spatial_unshared_input_projection,
        change_residual_mode=args.change_residual_mode,
        spatial_channel_attention=args.spatial_channel_attention,
        spatial_classifier_mode=args.spatial_classifier_mode,
        disable_sobel_modulation=args.disable_sobel_modulation,
        disable_change_residual_gate=args.disable_change_residual_gate,
        disable_mamba_module=args.disable_mamba_module,
        disable_vrwkv_module=args.disable_vrwkv_module,
        disable_pixel_spectral_grouping=args.disable_pixel_spectral_grouping,
        spectral_branch_only=args.spectral_branch_only_ablation,
        raw_center_difference_only=args.raw_center_difference_only,
        spatial_branch_only=args.spatial_branch_only_ablation,
        disable_pixel_global_pooling=args.disable_pixel_global_pooling,
        disable_pixel_global_component=args.disable_pixel_global_component,
        include_pixel_ungrouped_component=args.include_pixel_ungrouped_component,
        pixel_spectral_remove_shared_encoder=args.pixel_spectral_remove_shared_encoder,
        pixel_spectral_remove_mean=args.pixel_spectral_remove_mean,
        pixel_spectral_raw_grouping=args.pixel_spectral_raw_grouping,
        pixel_spectral_adaptive_multiscale=
        args.pixel_spectral_adaptive_multiscale,
        pixel_spectral_dense_kernel9=args.pixel_spectral_dense_kernel9,
        spectral_inter_only=args.spectral_inter_only,
        context_pool_mode=args.context_pool_mode,
        context_temporal_weight=args.context_temporal_weight,
        stable_residual_init=args.stable_residual_init,
        scene_codebook_size=args.scene_codebook_size,
        scene_prior_dim=args.scene_prior_dim,
        scene_reliability_gate=args.scene_reliability_gate,
        pixel_spectral_groups=args.pixel_spectral_groups,
        pixel_spectral_width=args.pixel_spectral_width,
        pixel_spectral_dim=args.pixel_spectral_dim,
        pixel_spectral_linear_length=args.pixel_spectral_linear_length,
        pixel_spectral_kernel=args.pixel_spectral_kernel,
        pixel_fusion_alpha=args.pixel_fusion_alpha,
        pixel_spectral_design=args.pixel_spectral_design,
        pixel_adaptive_fusion=args.pixel_adaptive_fusion,
        pixel_independent_fusion=args.pixel_independent_fusion,
        pixel_dual_reliability_fusion=args.pixel_dual_reliability_fusion,
        pixel_complementarity_fusion=
        args.pixel_complementarity_fusion,
        pixel_complementarity_v2_fusion=
        args.pixel_complementarity_v2_fusion,
        pixel_complementarity_v3_fusion=
        args.pixel_complementarity_v3_fusion,
        pixel_complementarity_v4_fusion=
        args.pixel_complementarity_v4_fusion,
        pixel_complementarity_v5_fusion=
        args.pixel_complementarity_v5_fusion,
        pixel_complementarity_v6_fusion=
        args.pixel_complementarity_v6_fusion,
        pixel_complementarity_v7_fusion=
        args.pixel_complementarity_v7_fusion,
        pixel_complementarity_v8_fusion=
        args.pixel_complementarity_v8_fusion,
        pixel_complementarity_v9_fusion=
        args.pixel_complementarity_v9_fusion,
        pixel_complementarity_v10_fusion=
        args.pixel_complementarity_v10_fusion,
        pixel_complementarity_v11_fusion=
        args.pixel_complementarity_v11_fusion,
        pixel_batch_scene_fusion=args.pixel_batch_scene_fusion,
        pixel_batch_full_v11_fusion=args.pixel_batch_full_v11_fusion,
        spectral_only_classifier=args.spectral_only_classifier,
        spectral_difference_only=args.spectral_difference_only,
        patch_difference_only=args.patch_difference_only,
        pixel_complementarity_v12_fusion=
        args.pixel_complementarity_v12_fusion,
        pixel_complementarity_v13_fusion=
        args.pixel_complementarity_v13_fusion,
        pixel_complementarity_v14_fusion=
        args.pixel_complementarity_v14_fusion,
        pixel_two_statistic_fusion=args.pixel_two_statistic_fusion,
        pixel_geomean_fusion=args.pixel_geomean_fusion,
        pixel_decomposed_context_fusion=
        args.pixel_decomposed_context_fusion,
        pixel_two_statistic_direction_power=
        args.pixel_two_statistic_direction_power,
        pixel_two_statistic_slope=args.pixel_two_statistic_slope,
        pixel_two_statistic_center=args.pixel_two_statistic_center)
    model.local_only_views = args.local_only_views
    model.disable_context_modeling = args.disable_context_modeling
    model.embedded_reliability_margin = args.physics_margin
    model.learned_fusion_min_confidence = args.learned_fusion_min_confidence
    model.learned_fusion_conflict_margin = args.learned_fusion_conflict_margin
    model.disable_embedded_physics_routing = args.disable_embedded_physics_routing
    model.disable_embedded_physics_branch = args.disable_embedded_physics_branch
    model.disable_learned_difference_routing = \
        args.disable_learned_difference_routing
    model.coarse_ablation_mode = args.coarse_ablation_mode
    model.single_pass = args.single_pass
    model.pixel_main_loss_weight = args.pixel_main_loss_weight
    model.pixel_branch_loss_weight = args.pixel_branch_loss_weight
    model.pixel_fused_loss_weight = args.pixel_fused_loss_weight
    model.pixel_reliability_loss_weight = args.pixel_reliability_loss_weight
    spatial_modes = sum((args.four_cross_spatial,
                         args.horizontal_bidirectional_spatial,
                         args.horizontal_forward_only_spatial,
                         args.horizontal_vertical_forward_spatial))
    if spatial_modes > 1:
        raise ValueError("spatial cross modes are mutually exclusive")
    if args.four_cross_spatial:
        if not hasattr(model.spatial_model, "four_cross"):
            raise ValueError("four-cross spatial requires CrossSpatialVRWKV")
        model.spatial_model.four_cross = True
    if args.horizontal_bidirectional_spatial:
        if not hasattr(model.spatial_model, "horizontal_bidirectional"):
            raise ValueError("horizontal bidirectional mode requires CrossSpatialVRWKV")
        model.spatial_model.horizontal_bidirectional = True
    if args.horizontal_forward_only_spatial:
        if not hasattr(model.spatial_model, "horizontal_forward_only"):
            raise ValueError("horizontal forward-only mode requires CrossSpatialVRWKV")
        model.spatial_model.horizontal_forward_only = True
    if args.horizontal_vertical_forward_spatial:
        if not hasattr(model.spatial_model, "horizontal_vertical_forward"):
            raise ValueError("horizontal-vertical forward mode requires CrossSpatialVRWKV")
        model.spatial_model.horizontal_vertical_forward = True
    if args.learn_axis_balance:
        if not args.four_cross_spatial:
            raise ValueError("learn-axis-balance requires four-cross-spatial")
        model.spatial_model.learn_axis_balance = True
    model = model.cuda()
    if embedded_calibrator is not None:
        pipeline = embedded_calibrator.grouped.named_steps
        scaler = pipeline["standardscaler"]
        logistic = pipeline["logisticregression"]
        model.set_embedded_logistic(
            torch.from_numpy(scaler.mean_).float().cuda(),
            torch.from_numpy(scaler.scale_).float().cuda(),
            torch.from_numpy(logistic.coef_).float().cuda(),
            torch.from_numpy(logistic.intercept_).float().cuda(),
            embedded_calibrator.decision_threshold)
    init_audit = None
    if args.init_state:
        init_path = Path(args.init_state)
        if not init_path.exists():
            candidate = root.parent / init_path
            if candidate.exists():
                init_path = candidate
        saved_state = torch.load(init_path, map_location="cpu")
        current_state = model.state_dict()
        compatible_state = {
            name: value for name, value in saved_state.items()
            if name in current_state
            and current_state[name].shape == value.shape
            and not name.endswith("spatial_decay")
        }
        missing, unexpected = model.load_state_dict(
            compatible_state, strict=False)
        allowed_missing = {"sobel.kx", "sobel.ky",
                           "spatial_model.axis_balance_logit",
                           "input_projection.weight",
                           "reliability_feature_mean",
                           "reliability_feature_scale"}
        allowed_prefixes = ("embedded_physics_head.",
                            "learned_difference_encoder.",
                            "learned_difference_head.")
        unsupported_missing = {
            name for name in missing
            if name not in allowed_missing
            and not name.endswith("spatial_decay")
            and not name.startswith(allowed_prefixes)
        }
        if unsupported_missing or unexpected:
            raise RuntimeError(
                f"incompatible initialization: missing={sorted(unsupported_missing)}, "
                f"unexpected={unexpected}")
        init_audit = {"path": str(init_path.resolve()),
                      "sha256": file_sha256(init_path)}
    # Negative values preserve the validated initialization.
    with torch.no_grad():
        if args.edge_init >= 0 and model.edge_scale is not None:
            model.edge_scale.fill_(args.edge_init)
        if args.detail_init >= 0 and model.detail_scale is not None:
            model.detail_scale.fill_(args.detail_init)
        if (args.change_residual_init >= 0
                and model.change_residual is not None):
            model.change_residual.scale.fill_(args.change_residual_init)
    label_smoothing = 0.02
    if args.label_smoothing >= 0:
        label_smoothing = args.label_smoothing
    class_weight = torch.tensor(
        [args.change_class_weight, 1.0], dtype=torch.float32).cuda()
    if args.focal_gamma > 0:
        criterion = FocalCrossEntropy(
            class_weight, label_smoothing, args.focal_gamma).cuda()
    else:
        criterion = torch.nn.CrossEntropyLoss(
            weight=class_weight, label_smoothing=label_smoothing).cuda()
    decay, regular, fusion_parameters = [], [], []
    for parameter_name, parameter in model.named_parameters():
        if parameter_name in {"pixel_main_scale_raw",
                              "pixel_spectral_scale_raw"}:
            fusion_parameters.append(parameter)
        else:
            (decay if "spatial_decay" in parameter_name else regular).append(
                parameter)
    parameter_groups = [
        {"params": regular, "weight_decay": args.weight_decay},
        {"params": decay, "weight_decay": 0.0,
         "lr": args.lr * args.rwkv_lr_multiplier}]
    if fusion_parameters:
        parameter_groups.append({
            "params": fusion_parameters, "weight_decay": 0.0,
            "lr": args.lr * args.fusion_lr_multiplier})
    optimizer = torch.optim.AdamW(
        parameter_groups, lr=args.lr, amsgrad=not args.disable_amsgrad)
    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.epochs, eta_min=args.lr * args.lr_min_ratio)
    elif args.lr_scheduler == "constant":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda _: 1.0)
    elif args.lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, args.lr_step_size), gamma=args.lr_gamma)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=sorted(set(args.lr_milestones)),
            gamma=args.lr_gamma)
    sam_rho = max(args.sam_rho, 0.0)
    prototype_memory = None
    if args.prototype_memory_weight > 0:
        codebook_size = max(1, args.prototype_codebook_size)
        center_shape = ((2, model.feature_dim) if codebook_size == 1
                        else (2, codebook_size, model.feature_dim))
        initialized_shape = ((2,) if codebook_size == 1
                             else (2, codebook_size))
        prototype_memory = {
            "centers": torch.zeros(*center_shape, device="cuda"),
            "initialized": torch.zeros(*initialized_shape, dtype=torch.bool,
                                       device="cuda"),
            "momentum": args.prototype_memory_momentum,
            "temperature": args.prototype_codebook_temperature}
    # EMA is a training-only variance reduction track: it changes neither the
    # architecture nor the number of inference parameters.  Updating once per
    # epoch makes 0.90 average roughly the most recent ten epochs.
    ema_decay = args.model_ema_decay
    ema = ({name: value.detach().clone()
            for name, value in model.state_dict().items()}
           if ema_decay > 0 else None)
    ema_updates = 0
    records = []
    snapshot_probabilities = []
    snapshot_ensemble = None
    best_validation_key = None
    best_validation_state = None
    best_validation_alpha = None
    best_validation_rule = None

    def emit(message):
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    if log_path.exists():
        raise FileExistsError(f"refusing to append to existing log: {log_path}")
    emit("CONFIG " + json.dumps(vars(args), sort_keys=True))
    emit(f"SPLIT dataset={args.dataset} bands={split.first.shape[2]} "
         f"labelled_pool={len(split.train_labels)} "
         f"fit={len(fit_labels)} "
         f"changed={int((fit_labels == 0).sum())} "
         f"unchanged={int((fit_labels == 1).sum())} "
         f"validation={0 if validation_labels is None else len(validation_labels)} "
         f"validation_changed={0 if validation_labels is None else int((validation_labels == 0).sum())} "
         f"validation_unchanged={0 if validation_labels is None else int((validation_labels == 1).sum())} "
         f"test={len(evaluation_test_labels)} "
         f"test_original={len(split.test_labels)} "
         f"test_excluded={len(split.test_labels) - len(evaluation_test_labels)} "
         f"test_exclusion_radius={args.test_exclusion_radius} "
         f"changed={int((evaluation_test_labels == 0).sum())} "
         f"unchanged={int((evaluation_test_labels == 1).sum())}")
    emit("PHYSICS " + json.dumps(audit, sort_keys=True))
    emit("REPRODUCIBILITY " + json.dumps({
        "split_sha256": file_sha256(split_path),
        "selection_split_sha256": file_sha256(selection_path),
        "source_sha256": {
            name: file_sha256(root / name) for name in (
                "train.py", "data.py", "physics.py", "core/model.py")
        }}, sort_keys=True))
    if init_audit is not None:
        emit("INITIALIZATION " + json.dumps(init_audit, sort_keys=True))
    pixel_prototype_audit = refresh_pixel_training_prototypes(
        model, train_eval_loader)
    if pixel_prototype_audit is not None:
        emit("PIXEL_PROTOTYPE_INITIALIZE " + json.dumps(
            pixel_prototype_audit, sort_keys=True))
    for epoch in range(1, args.epochs + 1):
        if (prototype_memory is not None
                and args.prototype_global_bootstrap
                and epoch == args.prototype_memory_delay + 1):
            bootstrap_prototype_memory(model, train_eval_loader,
                                       prototype_memory)
            emit(f"MEMORY_BOOTSTRAP Epoch: {epoch} | "
                 f"samples: {len(fit_labels)}")
        memory_weight = args.prototype_memory_weight
        if prototype_memory is not None:
            if epoch <= args.prototype_memory_delay:
                memory_weight = 0.0
            elif args.prototype_memory_warmup > 0:
                ramp_epoch = epoch - args.prototype_memory_delay
                memory_weight *= min(1.0, ramp_epoch / args.prototype_memory_warmup)
        prototype_weight = args.prototype_weight
        if args.prototype_decay_start > 0 and epoch >= args.prototype_decay_start:
            remaining = max(0, args.epochs - epoch)
            duration = max(1, args.epochs - args.prototype_decay_start)
            prototype_weight *= remaining / duration
        auxiliary_scale = 1.0
        if args.aux_decay_start > 0 and epoch >= args.aux_decay_start:
            remaining = max(0, args.epochs - epoch)
            duration = max(1, args.epochs - args.aux_decay_start)
            auxiliary_scale = remaining / duration
        spectral_style_scale = 1.0
        if args.spectral_style_warmup > 0:
            spectral_style_scale = min(
                1.0, epoch / float(args.spectral_style_warmup))
        train = train_epoch(model, train_loader, optimizer, scheduler,
                            criterion, sam_rho, prototype_weight,
                            prototype_memory, memory_weight,
                            args.spatial_scale_aug, args.hard_example_weight,
                            args.hard_example_fraction,
                            args.same_class_mix_prob,
                            args.same_class_mix_strength,
                            args.spectral_group_mask_prob,
                            args.spectral_group_mask_fraction,
                            args.spectral_gain_prob,
                            args.spectral_gain_strength * spectral_style_scale,
                            args.spectral_offset_prob,
                            args.spectral_offset_strength * spectral_style_scale,
                            args.spatial_illumination_prob,
                            args.spatial_illumination_strength,
                            args.temporal_feature_weight,
                            args.boundary_loss_weight,
                            args.supervised_contrastive_weight,
                            args.midpoint_style_prob,
                            args.midpoint_style_strength,
                            args.swap_consistency_weight,
                            args.vat_weight,
                            args.coarse_aux_weight * auxiliary_scale,
                            args.context_aux_weight * auxiliary_scale,
                            args.embedded_physics_aux_weight * auxiliary_scale,
                            args.learned_difference_aux_weight * auxiliary_scale)
        # Refresh after every epoch from an ordered pass over all optimizer
        # training samples. Validation and test loaders are never passed here.
        pixel_prototype_audit = refresh_pixel_training_prototypes(
            model, train_eval_loader)
        if ema is not None and epoch >= args.model_ema_start:
            with torch.no_grad():
                for name, value in model.state_dict().items():
                    if ema_updates == 0:
                        ema[name].copy_(value.detach())
                    elif value.is_floating_point():
                        ema[name].lerp_(value.detach(), 1.0 - ema_decay)
                    else:
                        ema[name].copy_(value.detach())
            ema_updates += 1
        component_text = ""
        if train.get("fused_loss") is not None:
            component_text = (
                f" | fused_loss: {train['fused_loss']:.6f}"
                f" | vrwkv_loss: {train['vrwkv_loss']:.6f}"
                f" | spectral_loss: {train['spectral_loss']:.6f}")
        emit(f"TRAIN Epoch: {epoch:03d} | loss: {train['loss']:.6f} | "
             f"OA: {train['oa']:.6f}{component_text}"
             f" | memory_weight: {memory_weight:.6f}")
        scheduled_test = (
            epoch % args.test_freq == 0 if args.test_at_round_multiples
            else (epoch - 1) % args.test_freq == 0)
        if not scheduled_test and epoch != args.epochs:
            continue
        if pixel_prototype_audit is not None:
            emit("PIXEL_PROTOTYPE_REFRESH " + json.dumps({
                "epoch": epoch, **pixel_prototype_audit,
                "mix": float(torch.sigmoid(
                    model.pixel_spectral_branch.prototype_mix_logit
                ).detach().cpu())}, sort_keys=True))
        backup = None
        if ema is not None and ema_updates > 0:
            backup = {name: value.detach().clone()
                      for name, value in model.state_dict().items()}
            with torch.no_grad():
                model.load_state_dict(ema, strict=True)
        train_eval = evaluated_metrics(
            model, train_eval_loader, criterion, calibrator, args.spatial_tta,
            include_predictions=(args.structure_fusion in
                                 ("scene", "scene_competence",
                                  "scene_train_structure")),
            physics_temporal_unreliable=args.physics_temporal_unreliable)
        gate_eval = (None if gate_loader is None else evaluated_metrics(
            model, gate_loader, criterion, calibrator, args.spatial_tta,
            include_predictions=True,
            physics_temporal_unreliable=args.physics_temporal_unreliable))
        validation_eval = (None if validation_loader is None else
            evaluated_metrics(
                model, validation_loader, criterion, calibrator,
                args.spatial_tta,
                include_predictions=(args.validation_scene_fusion
                                     or args.validation_reliability_fusion
                                     or args.staged_holdout_gate
                                     or args.structure_fusion ==
                                        "scene_train_structure"),
                physics_temporal_unreliable=args.physics_temporal_unreliable))
        validation_alpha = None
        validation_rule = None
        if args.staged_holdout_gate:
            if gate_eval is None or validation_eval is None:
                raise RuntimeError("staged gate requires gate and selection sets")
            if args.structure_fusion in ("scene", "scene_competence"):
                fit_rule = (fit_scene_competence_rule
                            if args.structure_fusion == "scene_competence"
                            else fit_scene_structure_rule)
                validation_rule = fit_rule(gate_eval, train_eval)
                validation_eval = apply_scene_structure_metrics(
                    validation_eval, validation_rule)
                for key in ("calibrated_probability", "target",
                            "main_probability", "pixel_probability",
                            "structure_feature"):
                    train_eval.pop(key, None)
            else:
                validation_rule = fit_staged_holdout_gate(
                    gate_eval, args.seed + epoch,
                    use_structure=(args.structure_fusion == "pixel"))
                validation_eval = apply_staged_gate_metrics(
                    validation_eval, validation_rule)
        if args.structure_fusion == "scene_train_structure":
            if validation_eval is None:
                raise RuntimeError("training-structure fusion needs validation")
            validation_rule = fit_train_structure_rule(train_eval)
            validation_eval = apply_scene_structure_metrics(
                validation_eval, validation_rule)
            for key in ("calibrated_probability", "target",
                        "main_probability", "pixel_probability",
                        "structure_feature"):
                train_eval.pop(key, None)
        if args.validation_scene_fusion:
            if validation_eval is None:
                raise RuntimeError(
                    "validation-scene-fusion has no validation samples")
            validation_alpha = fit_validation_scene_alpha(
                validation_eval,
                args.validation_fusion_min_alpha,
                args.validation_fusion_max_alpha,
                args.validation_fusion_shrinkage)
            validation_eval = apply_scene_fusion_metrics(
                validation_eval, validation_alpha)
        if args.validation_reliability_fusion:
            if validation_eval is None:
                raise RuntimeError(
                    "validation-reliability-fusion has no validation samples")
            validation_rule = fit_validation_reliability_rule(
                validation_eval, args.reliability_local_strength)
            validation_eval = apply_reliability_fusion_metrics(
                validation_eval, validation_rule)
        # A validation-selected protocol must not repeatedly inspect the test
        # set.  Keep only the best validation state in memory and evaluate that
        # state on the test set exactly once after training.
        if validation_eval is None:
            test_eval = evaluated_metrics(
                model, test_loader, criterion, calibrator, args.spatial_tta,
                include_predictions=args.snapshot_ensemble,
                physics_temporal_unreliable=args.physics_temporal_unreliable)
            test_probability = test_eval.pop("calibrated_probability", None)
            test_target = test_eval.pop("target", None)
        else:
            test_eval = None
            test_probability = None
            test_target = None
            validation_key = (validation_eval["calibrated"]["oa"],
                              -validation_eval["loss"])
            if best_validation_key is None or validation_key > best_validation_key:
                best_validation_key = validation_key
                best_validation_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()}
                best_validation_alpha = validation_alpha
                best_validation_rule = validation_rule
        if backup is not None:
            with torch.no_grad():
                model.load_state_dict(backup, strict=True)
        emit(format_metrics("TRAIN-EVAL RAW", epoch, train_eval["raw"]))
        if validation_eval is not None:
            emit(format_metrics("VALIDATION RAW", epoch,
                                validation_eval["raw"]))
            emit(format_metrics("VALIDATION CAL", epoch,
                                validation_eval["calibrated"]))
        if test_eval is not None:
            emit(format_metrics("TEST RAW", epoch, test_eval["raw"]))
            emit(format_metrics("TEST CAL", epoch, test_eval["calibrated"]))
            emit(f"AUDIT Epoch: {epoch:03d} | train_loss: {train_eval['loss']:.8f} "
                 f"| physics_overrides: {test_eval['physics_overrides']} "
                 f"| embedded_physics_overrides: "
                 f"{test_eval['embedded_physics_overrides']} "
                 f"| embedded_physics_override_rate: "
                 f"{test_eval['embedded_physics_override_rate']:.8f}")
        if args.snapshot_ensemble and epoch in {41, 51, 61}:
            snapshot_probabilities.append(test_probability)
            if epoch == 61:
                snapshot_ensemble = binary_metrics(
                    test_target,
                    np.mean(snapshot_probabilities, axis=0) >= 0.5)
                emit(format_metrics("SNAPSHOT ENSEMBLE", epoch,
                                    snapshot_ensemble))
        decay_state = {}
        for parameter_name, parameter in model.named_parameters():
            if ("spatial_decay_edge_scale" in parameter_name
                    or "spatial_decay_banks" in parameter_name):
                value = parameter.detach().float()
                decay_state[parameter_name] = {
                    "mean": float(value.mean()),
                    "min": float(value.min()), "max": float(value.max())}
        if decay_state:
            emit("DECAY_STATE " + json.dumps(
                {"epoch": epoch, "parameters": decay_state}, sort_keys=True))
        records.append({"epoch": epoch, "train": train,
                        "train_eval": train_eval,
                        "validation_eval": validation_eval,
                        "test_eval": test_eval})
        result_path.write_text(json.dumps({"config": vars(args), "records": records},
                                          indent=2), encoding="utf-8")
    train_selected = min(records, key=lambda row: (
        -row["train_eval"]["raw"]["oa"], row["train_eval"]["loss"]))
    if validation_loader is None:
        raw_test_peak = max(records, key=lambda row: row[
            "test_eval"]["raw"]["oa"])
        test_peak = max(records, key=lambda row: row[
            "test_eval"]["calibrated"]["oa"])
        selected = test_peak
    else:
        selected = max(records, key=lambda row: (
            row["validation_eval"]["calibrated"]["oa"],
            -row["validation_eval"]["loss"]))
        if best_validation_state is None:
            raise RuntimeError("validation protocol did not produce a checkpoint")
        model.load_state_dict(best_validation_state, strict=True)
        final_test = evaluated_metrics(
            model, test_loader, criterion, calibrator, args.spatial_tta,
            include_predictions=True,
            physics_temporal_unreliable=args.physics_temporal_unreliable,
            fusion_alpha_override=(
                args.test_fusion_alpha_override
                if args.test_fusion_alpha_override is not None
                else best_validation_alpha),
            fusion_rule_override=best_validation_rule)
        final_test_loose99 = evaluated_metrics(
            model, loose_test_loader, criterion, calibrator, args.spatial_tta,
            include_predictions=True,
            physics_temporal_unreliable=args.physics_temporal_unreliable,
            fusion_alpha_override=(
                args.test_fusion_alpha_override
                if args.test_fusion_alpha_override is not None
                else best_validation_alpha),
            fusion_rule_override=best_validation_rule)
        visualization_root = Path(os.environ.get(
            "VRWKV_VIS_ROOT", result_root / "visualization_inputs"))
        visualization_root.mkdir(parents=True, exist_ok=True)
        visualization_path = visualization_root / f"{name}.npz"
        np.savez_compressed(
            visualization_path,
            dataset=np.asarray(args.dataset), seed=np.int64(args.seed),
            selected_epoch=np.int64(selected["epoch"]),
            image_shape=np.asarray(split.labels.shape, dtype=np.int64),
            scene_labels=np.asarray(split.labels),
            train_positions=np.asarray(fit_positions, dtype=np.int64),
            train_labels=np.asarray(fit_labels, dtype=np.int64),
            validation_positions=np.asarray(validation_positions, dtype=np.int64),
            validation_labels=np.asarray(validation_labels, dtype=np.int64),
            loose_positions=np.asarray(split.test_positions, dtype=np.int64),
            loose_targets=np.asarray(final_test_loose99["target"], dtype=np.int64),
            loose_fused_probability=np.asarray(
                final_test_loose99["calibrated_probability"], dtype=np.float32),
            loose_spatial_probability=np.asarray(
                final_test_loose99["main_probability"], dtype=np.float32),
            loose_spectral_probability=np.asarray(
                final_test_loose99["pixel_probability"], dtype=np.float32),
            strict_positions=np.asarray(evaluation_test_positions, dtype=np.int64),
            strict_targets=np.asarray(final_test["target"], dtype=np.int64),
            strict_fused_probability=np.asarray(
                final_test["calibrated_probability"], dtype=np.float32),
            strict_spatial_probability=np.asarray(
                final_test["main_probability"], dtype=np.float32),
            strict_spectral_probability=np.asarray(
                final_test["pixel_probability"], dtype=np.float32),
            strict_keep_mask=np.asarray(evaluation_test_keep, dtype=np.bool_),
            test_exclusion_radius=np.int64(args.test_exclusion_radius),
            patch=np.int64(args.patch))
        for evaluated in (final_test, final_test_loose99):
            for key in ("calibrated_probability", "target",
                        "main_probability", "pixel_probability",
                        "structure_feature"):
                evaluated.pop(key, None)
        selected["test_eval"] = final_test
        selected["test_eval_loose99"] = final_test_loose99
        selected["visualization_input"] = str(visualization_path)
        emit(format_metrics("FINAL TEST RAW", selected["epoch"],
                            final_test["raw"]))
        emit(format_metrics("FINAL TEST CAL", selected["epoch"],
                            final_test["calibrated"]))
        emit(f"FINAL TEST AUDIT | physics_overrides: "
             f"{final_test['physics_overrides']} | embedded_physics_overrides: "
             f"{final_test['embedded_physics_overrides']} | "
             f"embedded_physics_override_rate: "
             f"{final_test['embedded_physics_override_rate']:.8f} | "
             f"corrected_to_right: {final_test['embedded_corrected_to_right']} | "
             f"corrected_to_wrong: {final_test['embedded_corrected_to_wrong']} | "
             f"corrected_unchanged: {final_test['embedded_corrected_unchanged']} | "
             f"main_uncorrected_oa: {final_test['main_uncorrected']['oa']:.8f} | "
             f"physics_routed_oa: {final_test['physics_routed']['oa']:.8f}")
        raw_test_peak = None
        test_peak = None
    summary = {"config": vars(args), "physics": audit, "records": records,
               "train_selected": train_selected,
               "raw_test_peak": raw_test_peak, "test_peak": test_peak,
               "protocol_selected": selected}
    if snapshot_ensemble is not None:
        summary["snapshot_ensemble"] = snapshot_ensemble
    result_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    emit("SUMMARY " + json.dumps({
        "selection_protocol": args.selection_protocol,
        "protocol_selected_epoch": selected["epoch"],
        "protocol_selected_validation": selected["validation_eval"],
        "protocol_selected_test_raw": selected["test_eval"]["raw"],
        "protocol_selected_test_cal": selected["test_eval"]["calibrated"],
        "train_selected_epoch": train_selected["epoch"],
        "train_selected_test_raw": (None if train_selected["test_eval"] is None
                                    else train_selected["test_eval"]["raw"]),
        "train_selected_test_cal": (None if train_selected["test_eval"] is None
                                    else train_selected["test_eval"]["calibrated"]),
        "raw_test_peak_epoch": (None if raw_test_peak is None else
                                raw_test_peak["epoch"]),
        "raw_test_peak": (None if raw_test_peak is None else
                          raw_test_peak["test_eval"]["raw"]),
        "test_peak_epoch": (None if test_peak is None else test_peak["epoch"]),
        "test_peak_cal": (None if test_peak is None else
                          test_peak["test_eval"]["calibrated"]),
        "snapshot_ensemble": snapshot_ensemble}, sort_keys=True))


if __name__ == "__main__":
    main()
