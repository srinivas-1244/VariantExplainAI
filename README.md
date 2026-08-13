# VariantExplainAI

## An Explainability-Driven k-Mer CNN-Sparse Transformer Framework for ClinVar-Based Pathogenicity Prediction

VariantExplainAI is a research-oriented deep learning framework for **allele-specific genomic variant pathogenicity prediction** using ClinVar data.

The framework combines:

* Paired reference/alternate haplotype modeling
* Overlapping k-mer sequence representation
* Convolutional motif extraction
* Sparse Transformer-based long-range contextual learning
* Variant-aware feature fusion
* Integrated Gradients
* Attention Rollout
* Explanation-alignment regularization
* Perturbation-based explanation-faithfulness evaluation

The proposed model, **VarMotif-TransNet**, is designed to distinguish pathogenic and benign genomic variants while explicitly preserving the molecular consequence of the **REF-to-ALT allele transformation**.

---

## Table of Contents

* [Overview](#overview)
* [Proposed Architecture](#proposed-architecture)
* [Main Contributions](#main-contributions)
* [Repository Structure](#repository-structure)
* [Dataset](#dataset)
* [Required Genomic Resources](#required-genomic-resources)
* [Data Preprocessing](#data-preprocessing)
* [Reference and Alternate Haplotype Construction](#reference-and-alternate-haplotype-construction)
* [k-mer Tokenization](#k-mer-tokenization)
* [Allele-Aware Representation](#allele-aware-representation)
* [VarMotif-TransNet](#varmotif-transnet)
* [Pathogenicity Classification](#pathogenicity-classification)
* [Training Objective](#training-objective)
* [Explainability](#explainability)
* [Training Configuration](#training-configuration)
* [Installation](#installation)
* [Configuration](#configuration)
* [Dataset Preparation](#dataset-preparation)
* [Training](#training)
* [Model Outputs](#model-outputs)
* [Performance Metrics](#performance-metrics)
* [Baseline Experiments](#baseline-experiments)
* [Ablation Experiments](#ablation-experiments)
* [Gene-Wise Generalization](#gene-wise-generalization)
* [Temporal Generalization](#temporal-generalization)
* [Class-Imbalance Experiments](#class-imbalance-experiments)
* [Explainability Analysis](#explainability-analysis)
* [Perturbation Faithfulness](#perturbation-faithfulness)
* [Statistical Testing](#statistical-testing)
* [Variant Subgroup Evaluation](#variant-subgroup-evaluation)
* [Comparison with Established Predictors](#comparison-with-established-predictors)
* [Reproducibility](#reproducibility)
* [Synthetic Smoke Testing](#synthetic-smoke-testing)
* [Recommended Experimental Workflow](#recommended-experimental-workflow)
* [Hardware](#hardware)
* [Important Scientific Note](#important-scientific-note)
* [Citation](#citation)
* [License](#license)
* [Disclaimer](#disclaimer)

---

# Overview

VariantExplainAI provides an explainability-driven deep learning pipeline for genomic variant pathogenicity prediction.

The framework processes both **reference and alternate genomic haplotypes** so that the model can learn the specific effect of a genomic alteration rather than simply memorizing genomic locations.

### Key Features

* Allele-specific representation
* Separate REF and ALT haplotypes
* Explicit allele-difference encoding
* Overlapping 5-mer tokenization
* Shared trainable k-mer embeddings
* Variant metadata and positional encoding
* Two-layer 1D-CNN motif extraction
* Sparse multi-head self-attention
* Local and strided-global contextual modeling
* Variant-aware global attention
* Binary pathogenicity classification
* Integrated Gradients explanations
* Attention Rollout
* Explanation-alignment regularization
* Perturbation-based explanation validation

---

# Proposed Architecture

The complete VariantExplainAI processing pipeline is:

```text
ClinVar Variant
      │
      ▼
Variant Filtering and Normalization
      │
      ▼
GRCh38 Reference Validation
      │
      ▼
Reference and Alternate Haplotype Construction
      │
      ▼
Overlapping k-mer Tokenization
      │
      ▼
Shared k-mer Embeddings
      │
      ├───────────────┐
      ▼               ▼
E_ref             E_alt
      │               │
      └───────┬───────┘
              ▼
      D = E_alt - E_ref
              │
              ▼
Variant Metadata + Positional Information
              │
              ▼
Allele-Aware Feature Fusion
              │
              ▼
Two-Layer 1D-CNN
              │
              ▼
Sparse Transformer Encoder
              │
              ▼
Contextual CLS Representation
              │
              ▼
Pathogenicity Classification
              │
              ▼
Benign / Pathogenic Probability
```

The framework additionally generates token-level explanations using **Integrated Gradients** and **Attention Rollout**.

---

# Main Contributions

### 1. Allele-Specific Representation

Reference and alternate haplotypes are constructed separately so that variants occurring at the same genomic locus but having different alternate alleles remain distinguishable.

### 2. Allele-Difference Encoding

The framework explicitly calculates:

```text
D = E_alt - E_ref
```

This representation captures the sequence changes introduced by the alternate allele.

### 3. CNN-Based Motif Learning

Two one-dimensional convolutional layers capture short-range allele-dependent genomic motifs.

### 4. Sparse Long-Range Attention

The Transformer combines:

* Local attention
* Strided global attention
* CLS-token global connectivity
* Variant-overlapping global connectivity

### 5. Explainability-Guided Learning

Explanation alignment encourages attribution patterns to remain consistent with biologically relevant variant-centered regions.

### 6. Independent Explanation Validation

Model explanations can be evaluated through:

* Biological annotation overlap
* Attribution stability
* Controlled perturbation experiments

---

# Repository Structure

```text
VariantExplainAI/
│
├── configs/
│   └── config.yaml
│
├── data/
│   ├── raw/
│   │   ├── clinvar/
│   │   ├── grch38/
│   │   └── gencode/
│   │
│   ├── processed/
│   └── splits/
│
├── src/
│   ├── data/
│   ├── models/
│   ├── explainability/
│   ├── training/
│   ├── evaluation/
│   └── utils/
│
├── experiments/
│   ├── train_main.py
│   ├── train_baselines.py
│   ├── run_ablation.py
│   ├── run_gene_holdout.py
│   ├── run_temporal_holdout.py
│   ├── run_imbalance.py
│   ├── run_explainability.py
│   └── run_perturbation.py
│
├── outputs/
│
├── tests/
│
├── requirements.txt
├── IMPLEMENTATION_NOTES.md
└── README.md
```

## Directory Description

| Directory / File          | Purpose                                                                      |
| ------------------------- | ---------------------------------------------------------------------------- |
| `configs/`                | Experiment and model configuration                                           |
| `configs/config.yaml`     | Main configuration for sequence processing, model architecture, and training |
| `data/raw/clinvar/`       | Raw ClinVar resources                                                        |
| `data/raw/grch38/`        | GRCh38 reference genome resources                                            |
| `data/raw/gencode/`       | GENCODE GRCh38 annotation resources                                          |
| `data/processed/`         | Preprocessed and normalized datasets                                         |
| `data/splits/`            | Train, validation, and test partitions                                       |
| `src/data/`               | Data loading, filtering, normalization, tokenization, and preprocessing      |
| `src/models/`             | VarMotif-TransNet and related models                                         |
| `src/explainability/`     | Integrated Gradients, Attention Rollout, and attribution utilities           |
| `src/training/`           | Training loops, losses, optimization, checkpointing, and scheduling          |
| `src/evaluation/`         | Evaluation metrics and analysis                                              |
| `src/utils/`              | Shared utility functions                                                     |
| `experiments/`            | Training and experimental scripts                                            |
| `outputs/`                | Checkpoints, predictions, metrics, figures, and logs                         |
| `tests/`                  | Software and pipeline tests                                                  |
| `requirements.txt`        | Python dependencies                                                          |
| `IMPLEMENTATION_NOTES.md` | Additional implementation notes                                              |
| `README.md`               | Project documentation                                                        |

---

# Dataset

VariantExplainAI is primarily designed for variants obtained from **ClinVar**.

The analytical workflow supports the following clinical-significance classes:

```text
Pathogenic
Likely Pathogenic
Benign
Likely Benign
```

For binary classification:

```text
Pathogenic = Pathogenic + Likely Pathogenic

Benign = Benign + Likely Benign
```

## Supported Variant Types

* Single nucleotide variants (SNVs)
* Short insertions
* Short deletions

Large structural variants and unresolved genomic variants should be excluded before sequence modeling.

---

# Required Genomic Resources

## ClinVar

Required information includes:

* Variant coordinates
* REF/ALT alleles
* Clinical significance
* Review status
* Variant identifiers

## GRCh38

Used for:

* Reference-allele validation
* Sequence-window extraction
* Reference haplotype generation

## GENCODE GRCh38

Used for:

* Gene annotation
* Coding-region identification
* Splice-region annotation
* Non-coding context annotation

---

# Data Preprocessing

The preprocessing pipeline is:

```text
ClinVar Records
      │
      ▼
Target Clinical Significance Filtering
      │
      ▼
Conflict Removal
      │
      ▼
Review Status Filtering
      │
      ▼
SNV / Short-Indel Selection
      │
      ▼
Multi-Allelic Decomposition
      │
      ▼
Variant Normalization
      │
      ▼
GRCh38 REF Validation
      │
      ▼
Duplicate Removal
      │
      ▼
Binary Label Mapping
```

Each normalized variant should have a unique identifier:

```text
chromosome-position-reference-alternate
```

---

# Reference and Alternate Haplotype Construction

For every variant, approximately **±1000 bp** of sequence context is extracted from GRCh38.

Two allele-specific sequences are constructed:

```text
S_ref
S_alt
```

### Reference Haplotype

`S_ref` contains the original reference allele.

### Alternate Haplotype

`S_alt` is generated by applying the normalized REF-to-ALT transformation.

For insertions and deletions, cropping or padding is applied to maintain consistent sequence length.

Additional variant information includes:

```text
Variant Type
REF Length
ALT Length
ΔL = |ALT| - |REF|
Variant Position Indicator
```

---

# k-mer Tokenization

VariantExplainAI uses overlapping **5-mers** with:

```text
k = 5
stride = 1
```

The theoretical canonical nucleotide vocabulary contains:

```text
4^5 = 1024
```

possible 5-mers.

Additional reserved tokens can be used for ambiguous or padded positions.

The reference and alternate sequences use the same tokenizer and shared embedding layer.

---

# Allele-Aware Representation

The model generates separate embeddings:

```text
E_ref
E_alt
```

The explicit allele-difference representation is:

```text
D = E_alt - E_ref
```

These representations are combined with:

* Positional encoding
* Variant-position information
* Variant type
* REF length
* ALT length
* Allele-length difference

The combined representation is then projected into the model hidden dimension.

---

# VarMotif-TransNet

## Motif-Aware 1D-CNN

The CNN module contains:

```text
Conv1D
Kernel = 5
Filters = 128
        │
        ▼
BatchNorm
        │
        ▼
ReLU
        │
        ▼
Conv1D
Kernel = 3
Filters = 256
        │
        ▼
BatchNorm
        │
        ▼
ReLU
```

Same-length padding preserves genomic positional correspondence.

---

# Sparse Transformer Encoder

The Transformer configuration is:

| Parameter               |   Setting |
| ----------------------- | --------: |
| Transformer Layers      |         4 |
| Attention Heads         |         8 |
| Hidden Dimension        |       256 |
| Feed-Forward Dimension  |       512 |
| Local Attention Radius  | 64 tokens |
| Global Attention Stride | 32 tokens |
| Dropout                 |      0.20 |

The sparse attention mechanism combines local and global contextual information while reducing unnecessary dense attention computation.

---

# Pathogenicity Classification

The final classification pipeline is:

```text
CLS Representation
       │
       ▼
Linear Projection
       │
       ▼
Softmax
       │
       ├──► Benign Probability
       │
       └──► Pathogenic Probability
```

**Pathogenic** is treated as the positive class.

Default classification threshold:

```text
0.50
```

---

# Training Objective

The primary classification objective uses class-weighted cross-entropy.

The total loss is:

```text
L_total = L_cls + λ_align × L_align
```

where:

```text
λ_align = 0.20
```

* `L_cls` = pathogenicity classification loss
* `L_align` = explanation-alignment loss

---

# Explainability

VariantExplainAI uses two complementary explanation approaches.

## Integrated Gradients

Integrated Gradients is the primary attribution method.

It produces position-wise importance scores:

```text
a = [a1, a2, ..., am]
```

These scores represent the contribution of genomic token positions to the predicted class.

## Attention Rollout

Attention Rollout recursively aggregates Transformer attention matrices to estimate the influence of genomic positions on the final contextual representation.

Attention Rollout is used as a complementary explanation rather than a replacement for Integrated Gradients.

---

# Explanation Alignment

A variant-centered target mask is constructed around the normalized variant locus.

The attribution distribution is normalized and compared with the target mask using a cosine-distance alignment objective.

This regularization encourages the model to use biologically plausible variant-centered evidence while retaining the primary classification objective.

---

# Training Configuration

| Parameter               |  Setting |
| ----------------------- | -------: |
| Reference Context       | ±1000 bp |
| k-mer Size              |        5 |
| k-mer Stride            |        1 |
| Embedding Dimension     |      128 |
| Hidden Dimension        |      256 |
| CNN Filters             | 128, 256 |
| CNN Kernels             |     5, 3 |
| Transformer Layers      |        4 |
| Attention Heads         |        8 |
| FFN Dimension           |      512 |
| Local Attention Radius  |       64 |
| Global Attention Stride |       32 |
| Dropout                 |     0.20 |
| Optimizer               |    AdamW |
| Initial Learning Rate   |     1e-4 |
| Minimum Learning Rate   |     1e-6 |
| Weight Decay            |     1e-4 |
| Batch Size              |       64 |
| Maximum Epochs          |       80 |
| Early Stopping Patience |       10 |
| Gradient Clipping       |      1.0 |
| Alignment Loss Weight   |     0.20 |
| Random Seed             |       42 |

---

# Installation

## 1. Clone the Repository

```bash
git clone <repository-url>
cd VariantExplainAI
```

## 2. Create a Virtual Environment

```bash
python -m venv venv
```

### Linux / macOS

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Main Dependencies

The project uses:

* Python
* PyTorch
* NumPy
* Pandas
* scikit-learn
* SciPy
* Matplotlib
* PyYAML
* Captum

A CUDA-enabled NVIDIA GPU is strongly recommended for full-scale training.

---

# Configuration

Experiment parameters are maintained in:

```text
configs/config.yaml
```

Example configuration:

```yaml
seed: 42

sequence:
  context_radius: 1000
  kmer_size: 5
  stride: 1

model:
  embedding_dim: 128
  hidden_dim: 256
  transformer_layers: 4
  attention_heads: 8
  ffn_dim: 512
  dropout: 0.20
  local_radius: 64
  global_stride: 32

training:
  batch_size: 64
  max_epochs: 80
  learning_rate: 0.0001
  min_learning_rate: 0.000001
  weight_decay: 0.0001
  early_stopping_patience: 10
  gradient_clip: 1.0
  lambda_align: 0.20
```

Dataset locations and experiment parameters can be changed without modifying the main source code.

---

# Dataset Preparation

Place the required input resources under:

```text
data/raw/
├── clinvar/
├── grch38/
└── gencode/
```

Run the preprocessing workflow before training.

The preprocessing workflow performs:

1. ClinVar filtering
2. Variant normalization
3. GRCh38 validation
4. REF/ALT haplotype construction
5. k-mer preparation
6. Metadata encoding
7. Dataset partition generation

Processed data are stored under:

```text
data/processed/
```

Partition information is stored under:

```text
data/splits/
```

---

# Training VariantExplainAI

Run the primary model with:

```bash
python experiments/train_main.py
```

Training workflow:

```text
Dataset Loading
      ↓
Allele-Aware Representation
      ↓
VarMotif-TransNet Forward Propagation
      ↓
Classification Loss
      ↓
Explanation Alignment
      ↓
Backpropagation
      ↓
Validation AUROC Monitoring
      ↓
Best Checkpoint Selection
```

---

# Model Outputs

Generated files are stored under:

```text
outputs/
├── checkpoints/
├── predictions/
├── metrics/
├── figures/
└── logs/
```

The main prediction output contains fields such as:

```text
variant_id
true_label
predicted_label
p_benign
p_pathogenic
variant_type
gene
region
```

---

# Performance Metrics

VariantExplainAI supports:

* Accuracy
* Precision
* Recall
* Specificity
* F1-score
* AUROC
* AUPRC
* Confusion Matrix
* ROC Curve
* Precision-Recall Curve

The default threshold-dependent classification threshold is:

```text
0.50
```

AUROC and AUPRC are calculated from continuous pathogenicity probabilities.

---

# Baseline Experiments

Run:

```bash
python experiments/train_baselines.py
```

The baseline experiments include:

* Random Forest
* SVM-RBF
* CNN-only
* Transformer-only
* Sparse Transformer-only

Neural baselines use the same allele-aware sequence representation and stored data partitions whenever appropriate.

---

# Ablation Experiments

Run:

```bash
python experiments/run_ablation.py
```

The ablation study evaluates:

* Alternate allele representation
* Allele-difference encoding
* Variant metadata
* CNN motif extractor
* Sparse Transformer
* Sparse versus dense attention
* Explanation alignment
* Variant-global connectivity
* Strided-global connectivity

These experiments quantify the contribution of individual components of the proposed architecture.

---

# Gene-Wise Generalization

Run:

```bash
python experiments/run_gene_holdout.py
```

Under this protocol, variants belonging to the same gene are restricted to a single partition.

This evaluation determines whether the model generalizes to **unseen genes** instead of learning gene-specific sequence signatures.

---

# Temporal Generalization

Run:

```bash
python experiments/run_temporal_holdout.py
```

Earlier ClinVar data are used for model development, while newly introduced variants from a later release are reserved for testing.

This protocol approximates prospective pathogenicity prediction.

---

# Class-Imbalance Experiments

Run:

```bash
python experiments/run_imbalance.py
```

Pathogenic training ratios include:

```text
1:1
1:2
1:4
```

Only the training partition should be modified.

Validation and test distributions remain unchanged.

---

# Explainability Analysis

Run:

```bash
python experiments/run_explainability.py
```

The analysis includes:

* Integrated Gradients
* Attention Rollout
* Variant-centered attribution analysis
* Biological annotation overlap
* Top-k functional hit analysis
* Attribution stability

---

# Perturbation Faithfulness

Run:

```bash
python experiments/run_perturbation.py
```

Perturbation experiments include:

* Top-attribution masking
* Variant-centered masking
* Sequence-matched control masking
* Random distal masking
* REF/ALT perturbation

The confidence change is calculated as:

```text
ΔP = P(original) - P(perturbed)
```

Larger confidence drops indicate greater influence of the perturbed genomic region.

---

# Statistical Testing

Perturbation results can be compared using:

* Wilcoxon signed-rank test
* Holm multiple-comparison correction
* Confidence intervals for perturbation-based effect estimates

---

# Variant Subgroup Evaluation

The framework supports evaluation across:

* SNVs
* Short indels
* Coding variants
* Splice-region variants
* Non-coding variants

This allows model behavior to be assessed across different biological variant contexts.

---

# Comparison with Established Predictors

VariantExplainAI can support comparisons with established pathogenicity predictors, including:

* CADD
* REVEL
* ClinPred
* BayesDel
* PrimateAI
* EVE
* AlphaMissense
* SpliceAI

Comparisons should only be performed on variants supported by the corresponding external predictor.

The common eligible subset should therefore be used:

```text
VariantExplainAI predictions
            ∩
External predictor coverage
```

This prevents unfair comparisons involving unsupported variant classes.

---

# Reproducibility

VariantExplainAI emphasizes reproducible experimentation through:

* Fixed random seed
* Stored dataset partitions
* Configuration files
* Best-model checkpoints
* Saved predictions
* Experiment logs
* Independent evaluation scripts

Default random seed:

```text
42
```

The same stored partitions should be used for:

* Main model
* Baselines
* Ablations
* Explainability studies
* Robustness experiments

---

# Synthetic Smoke Testing

The repository includes a lightweight synthetic-data pathway for checking the software installation without requiring the complete ClinVar dataset.

The synthetic workflow can validate:

```text
Tokenization
     ↓
Dataset Loading
     ↓
Tensor Dimensions
     ↓
Model Forward Propagation
     ↓
Loss Computation
     ↓
Training
     ↓
Checkpoint Saving
     ↓
Metric Calculation
     ↓
Explainability Execution
```

> **Important:** Synthetic testing is intended only for software validation and should not be used for reporting scientific performance.

---

# Recommended Experimental Workflow

The recommended complete workflow is:

```text
1. Obtain ClinVar, GRCh38 and GENCODE resources
                ↓
2. Preprocess ClinVar records
                ↓
3. Normalize and validate variants
                ↓
4. Construct REF/ALT haplotypes
                ↓
5. Generate dataset splits
                ↓
6. Train VariantExplainAI
                ↓
7. Evaluate independent test performance
                ↓
8. Train internal baselines
                ↓
9. Perform ablation analysis
                ↓
10. Perform gene-wise holdout
                ↓
11. Perform temporal holdout
                ↓
12. Conduct class-imbalance experiments
                ↓
13. Generate Integrated Gradients explanations
                ↓
14. Generate Attention Rollout explanations
                ↓
15. Perform perturbation-faithfulness testing
                ↓
16. Conduct subgroup analysis
                ↓
17. Compare against external predictors
                ↓
18. Generate publication tables and figures
```

---

# Hardware

The intended full-scale implementation is suitable for CUDA-enabled GPU execution.

The research configuration was designed around a GPU-class workstation comparable to:

```text
NVIDIA RTX 3090
24 GB GPU Memory
```

Smaller configurations and synthetic experiments can be executed on lower-memory GPUs or CPU environments by reducing:

* Batch size
* Sequence dimensions
* Model dimensions

---

# Important Scientific Note

VariantExplainAI is intended to implement the methodology reproducibly.

Reported performance values from a research manuscript should **not** be assumed to reproduce automatically.

Performance can depend on:

* ClinVar release
* Filtering results
* Variant normalization
* Reference-genome consistency
* Training environment
* External predictor coverage
* Randomness
* Deterministic-operation support

Users should generate experimental results from their own processed dataset and retained test predictions.

---

# Implementation Philosophy

A central design principle of VariantExplainAI is:

> **The model should learn the effect of the specific REF-to-ALT transformation rather than merely memorizing genomic loci.**

Therefore, careful validation of the following should be completed before interpreting final model performance:

* Allele normalization
* Reference consistency
* Alternate-haplotype construction
* Dataset leakage

---

# Citation

If you use VariantExplainAI in academic work, please cite the corresponding research article.

```bibtex
@article{variantExplainAI,
  title   = {VariantExplainAI: An Explainability-Driven k-Mer CNN-Sparse Transformer Framework for ClinVar-Based Pathogenicity Prediction},
  author  = {Authors},
  journal = {Journal},
  year    = {Year}
}
```

> Replace the placeholder bibliographic information after publication.

---

# License

Add the appropriate project license before public release.

Common choices for academic research software include:

* MIT License
* Apache License 2.0
* BSD 3-Clause License

The selected license should be included in a separate:

```text
LICENSE
```

file.

---

# Disclaimer

VariantExplainAI is a **research and experimental framework** for genomic variant pathogenicity prediction.

The predictions and explanations generated by this software should not be considered medical diagnoses or clinical recommendations.

Users are responsible for validating datasets, experimental results, genomic references, and model outputs before using the framework in research or other applications.
