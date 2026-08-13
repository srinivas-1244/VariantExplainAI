# VariantExplainAI / VarMotif-TransNet

Functional research implementation of the paper methodology **VariantExplainAI: An Explainability-Driven k-Mer CNN–Sparse Transformer Framework for ClinVar-Based Pathogenicity Prediction**.

## Implemented methodology

- ClinVar VCF parsing for P/LP/B/LB small variants with review-status filtering.
- Allele normalization and unique `chr:pos:ref:alt` identifiers.
- GRCh38 reference validation.
- Paired fixed-length REF and ALT haplotype construction.
- Overlapping 5-mer tokenization with a shared vocabulary/embedding.
- Explicit allele-difference representation `D = E_alt - E_ref`.
- Variant type, REF length, ALT length, delta-length, variant-position and positional encoding.
- Two-layer 1D-CNN motif extractor (kernels 5 and 3).
- Memory-conscious local + strided-global sparse multi-head attention.
- Global access for CLS and variant-overlapping positions.
- Binary benign/pathogenic classification head.
- Class-weighted cross entropy; focal-loss utility for imbalance sensitivity.
- Explanation alignment using differentiable Integrated-Gradients approximation on the fused representation.
- Full evaluation-time Integrated Gradients and attention rollout.
- ROC/PR metrics, subgroup utilities, perturbation utilities and statistical tests.
- Variant-wise and gene-wise split generation.
- Random Forest and SVM-RBF allele-aware k-mer baselines.
- Common-eligible-subset external-predictor comparison utility.
- Synthetic smoke-test generator.

## Important reproducibility note

The paper states exact final ClinVar cohort counts. This implementation **does not fabricate or subsample records merely to force those counts**. It applies the stated filtering rules and writes preprocessing errors so that the actual fixed-release cohort can be audited. Full reproduction therefore requires the exact archived ClinVar input release and the matching GRCh38 FASTA used by the study.

The training-time alignment implementation uses a configurable low-step differentiable IG approximation (`ig_steps_train`, default 8) because exact 50-step Integrated Gradients in every mini-batch is prohibitively expensive. Evaluation uses a higher-step IG setting (`ig_steps_eval`, default 50). Both settings are explicit and reproducible.

## Installation

```bash
cd VariantExplainAI
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install -e .
```

For GPU training, install a PyTorch build matching the installed NVIDIA CUDA driver before `pip install -e .` if required by your environment.

## Expected real-data inputs

Place or pass paths to:

1. Fixed ClinVar VCF release used by the experiment.
2. GRCh38 FASTA with chromosome names compatible with ClinVar (`chr1`, ..., `chrX`, etc.).
3. Optional GENCODE annotations for downstream region/subgroup enrichment.

The core preparation command is:

```bash
python experiments/prepare_data.py \
  --config configs/config.yaml \
  --vcf /path/to/clinvar.vcf.gz \
  --fasta /path/to/GRCh38.fa \
  --out data/processed/variants.csv
```

This writes the processed sequence table, preprocessing-error table, and variant-wise/gene-wise split CSVs.

## Main model training

```bash
python experiments/train_main.py \
  --config configs/config.yaml \
  --train data/splits/variant_train.csv \
  --val data/splits/variant_val.csv \
  --test data/splits/variant_test.csv \
  --out outputs/main
```

Outputs include:

- `best.pt`
- `history.csv`
- `test_predictions.csv`
- `test_metrics.csv`
- ROC and precision-recall figures.

## Explainability

```bash
python experiments/explain.py \
  --config configs/config.yaml \
  --data data/splits/variant_test.csv \
  --checkpoint outputs/main/best.pt \
  --out outputs/main/explanations.csv
```

## Perturbation faithfulness

```bash
python experiments/run_perturbation.py \
  --config configs/config.yaml \
  --data data/splits/variant_test.csv \
  --checkpoint outputs/main/best.pt
```

## Classical baselines

```bash
python experiments/train_baselines.py \
  --train data/splits/variant_train.csv \
  --test data/splits/variant_test.csv \
  --out outputs/baselines
```

## External predictor comparison

Prepare a CSV containing `variant_id,score` for a predictor such as CADD/REVEL/ClinPred/BayesDel/AlphaMissense on its supported variants and run:

```bash
python experiments/compare_external_predictor.py \
  --variant-predictions outputs/main/test_predictions.csv \
  --external external_scores.csv \
  --name AlphaMissense \
  --score-column score \
  --out outputs/alpha_missense_comparison.csv
```

The script automatically evaluates only the common eligible subset.

## Synthetic end-to-end smoke test

No ClinVar download is required:

```bash
python experiments/make_synthetic_demo.py --out data/demo --n 120 --length 101
python experiments/train_main.py \
  --config configs/demo.yaml \
  --train data/demo/train.csv \
  --val data/demo/val.csv \
  --test data/demo/test.csv \
  --out outputs/demo
```

The synthetic dataset is for software verification only and must never be reported as a scientific result.

## Tests

```bash
pytest -q
```

## Main configuration corresponding to the manuscript

- Context: ±1000 bp
- k-mer size: 5, stride 1
- Embedding: 128
- Hidden dimension: 256
- CNN filters: 128, 256
- CNN kernels: 5, 3
- Transformer layers: 4
- Attention heads: 8
- FFN: 512
- Local radius: 64 tokens
- Global stride: 32 tokens
- Global access: CLS + variant-overlapping positions
- Dropout: 0.20
- AdamW, LR 1e-4, minimum LR 1e-6
- Weight decay: 1e-4
- Batch: 64
- Maximum epochs: 80
- Validation-AUROC early stopping patience: 10
- Gradient clipping: 1.0
- Alignment weight: 0.20
- Seed: 42

## Scope of external resources

Scores from established predictors are intentionally not bundled. Different tools have different supported variant domains and licensing/distribution conditions. The included common-subset comparator avoids extrapolating a predictor beyond variants for which a score is actually supplied.

## Additional paper experiments

Gene-disjoint training:

```bash
python experiments/run_gene_holdout.py
```

Temporal holdout requires a processed table containing a `first_seen_date` column derived from archived ClinVar releases:

```bash
python experiments/make_temporal_split.py --data processed_with_first_seen.csv --cutoff 2025-12-31
```

Imbalance training sets (validation/test remain unchanged):

```bash
python experiments/make_imbalance_splits.py --ratios 1:1 1:2 1:4
```

Example ablations:

```bash
python experiments/run_ablation.py --ablation no_alignment
python experiments/run_ablation.py --ablation no_alt_change
python experiments/run_ablation.py --ablation cnn_only
python experiments/run_ablation.py --ablation sparse_transformer_only
```

See `IMPLEMENTATION_NOTES.md` for a direct manuscript-section-to-code mapping and explicit decisions where the manuscript does not specify an implementation constant.
