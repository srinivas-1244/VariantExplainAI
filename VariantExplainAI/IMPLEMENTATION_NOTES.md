# Implementation-to-Manuscript Mapping

## Section 3.1 — ClinVar cohort and partitions
- `src/variant_explain_ai/data/clinvar.py`
- `src/variant_explain_ai/data/splits.py`
- `experiments/prepare_data.py`
- `experiments/make_temporal_split.py`
- `experiments/make_imbalance_splits.py`

## Section 3.2 — REF/ALT contexts
- `src/variant_explain_ai/data/haplotype.py`
- `src/variant_explain_ai/data/preprocess.py`

## Section 3.3 — 5-mer and allele-aware representation
- `src/variant_explain_ai/data/kmer.py`
- `src/variant_explain_ai/data/dataset.py`
- `VariantExplainAI.fuse_inputs()` in `models/model.py`

## Sections 3.4–3.6 — VarMotif-TransNet and classification
- `src/variant_explain_ai/models/model.py`
- `src/variant_explain_ai/models/sparse_attention.py`
- `src/variant_explain_ai/models/transformer.py`

## Section 3.7 — Explainability/alignment
- `src/variant_explain_ai/explainability/alignment.py`
- `src/variant_explain_ai/explainability/ig.py`
- `src/variant_explain_ai/explainability/rollout.py`
- `src/variant_explain_ai/explainability/perturbation.py`

## Sections 3.8–3.9 — Training and algorithm
- `configs/config.yaml`
- `src/variant_explain_ai/training/trainer.py`
- `src/variant_explain_ai/training/losses.py`
- `experiments/train_main.py`

## Sections 3.10–4 — Evaluation
- `src/variant_explain_ai/evaluation/metrics.py`
- `src/variant_explain_ai/evaluation/statistics.py`
- `src/variant_explain_ai/evaluation/explainability_metrics.py`
- `src/variant_explain_ai/evaluation/subgroups.py`
- `experiments/train_baselines.py`
- `experiments/run_ablation.py`
- `experiments/run_gene_holdout.py`
- `experiments/compare_external_predictor.py`
- `experiments/run_perturbation.py`

## Explicit implementation choices where the manuscript is underspecified
1. Training-time explanation regularization uses differentiable IG over the fused allele-aware representation with 8 configurable integration steps.
2. Evaluation-time IG defaults to 50 steps.
3. Variant-centered explanation target radius defaults to 10 k-mer positions and is configurable.
4. Short indels are limited to maximum allele length 50 bp in the VCF parser; this can be changed in code/config if the study uses another definition.
5. External predictor scores are not downloaded or redistributed. Users provide method-specific score files, and the comparator evaluates only common eligible variants.
6. Temporal splitting requires a `first_seen_date` field. This cannot be inferred reliably from a single fixed VCF and should be created from archived-release comparison metadata.
