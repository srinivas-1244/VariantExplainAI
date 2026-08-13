VariantExplainAI

An Explainability-Driven k-Mer CNN-Sparse Transformer Framework for ClinVar-Based Pathogenicity Prediction





VariantExplainAI is a research-oriented deep learning framework for allele-specific genomic variant pathogenicity prediction using ClinVar data.

Table of Contents

Overview

Proposed Architecture

Main Contributions

Repository Structure

Dataset

Required Genomic Resources

Data Preprocessing

Reference and Alternate Haplotype Construction

k-mer Tokenization

Allele-Aware Representation

VarMotif-TransNet

Pathogenicity Classification

Training Objective

Explainability

Explanation Alignment

Training Configuration

Installation

Main Dependencies

Configuration

Dataset Preparation

Training VariantExplainAI

Model Outputs

Performance Metrics

Baseline Experiments

Ablation Experiments

Gene-Wise Generalization

Temporal Generalization

Class-Imbalance Experiments

Explainability Analysis

Perturbation Faithfulness

Statistical Testing

Variant Subgroup Evaluation

Comparison with Established Predictors

Reproducibility

Synthetic Smoke Testing

Recommended Full Experimental Workflow

Hardware

Important Scientific Note

Implementation Philosophy

Citation

License

Disclaimer

1. Overview

VariantExplainAI is a research-oriented deep learning framework for allele-specific genomic variant pathogenicity prediction using ClinVar data. The implementation combines paired reference/alternate haplotype modeling, overlapping k-mer sequence representation, convolutional motif extraction, sparse transformer-based long-range contextual learning, and explainability-guided optimization.

The proposed model, VarMotif-TransNet, is designed to distinguish pathogenic and benign genomic variants while explicitly preserving the molecular consequence of the REF-to-ALT allele transformation.

Paired reference and alternate genomic haplotypes

Explicit REF-to-ALT allele-difference representation

Overlapping 5-mer tokenization

Shared trainable k-mer embeddings

Variant metadata and positional encoding

Two-layer 1D-CNN motif extractor

Sparse multi-head self-attention

Local and strided-global contextual modeling

Variant-aware global attention

Binary pathogenicity classification

Integrated Gradients-based attribution

Attention rollout

Explanation-alignment regularization

Perturbation-based explanation-faithfulness evaluation

2. Proposed Architecture

The complete processing pipeline is:

ClinVar Variant
↓
Variant Filtering and Normalization
↓
GRCh38 Reference Validation
↓
Reference and Alternate Haplotype Construction
↓
Overlapping k-mer Tokenization
↓
Shared k-mer Embeddings
↓
Reference Embedding E_ref
Alternate Embedding E_alt
Allele Difference D = E_alt - E_ref
↓
Variant Metadata + Positional Information
↓
Allele-Aware Feature Fusion
↓
Two-Layer 1D-CNN
↓
Sparse Transformer Encoder
↓
Contextual CLS Representation
↓
Pathogenicity Classification
↓
Benign / Pathogenic Probability

The framework additionally generates token-level explanations using Integrated Gradients and attention rollout.

3. Main Contributions



   **Allele-Specific Representation:** Reference and alternate haplotypes are constructed separately so that variants occurring at the same genomic locus but having different alternate alleles remain distinguishable.



   **Allele-Difference Encoding:** The model explicitly computes D = E_alt - E_ref to represent sequence changes introduced by the alternate allele.



   **CNN-Based Motif Learning:** Two one-dimensional convolutional layers capture short-range allele-dependent genomic motifs.



   **Sparse Long-Range Attention:** A custom sparse transformer combines local attention, strided global attention, CLS-token global connectivity, and variant-overlapping global connectivity.



   **Explainability-Guided Learning:** Explanation alignment encourages attribution patterns to remain consistent with biologically relevant variant-centered regions.



   **Independent Explanation Validation:** Model explanations can be evaluated using biological annotation overlap, attribution stability, and controlled perturbation experiments.

4. Repository Structure

The repository is organized into configuration, data, source-code, experiment, output, and testing components:

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
├── tests/
│
├── requirements.txt
├── IMPLEMENTATION_NOTES.md
└── README.md

Directory Description

Directory / File

Purpose

configs/

Stores experiment and model configuration files.

configs/config.yaml

Main configuration for sequence processing, model architecture, and training.

data/raw/clinvar/

Raw ClinVar variant resources.

data/raw/grch38/

GRCh38 reference genome resources.

data/raw/gencode/

GENCODE GRCh38 annotation resources.

data/processed/

Preprocessed and normalized datasets.

data/splits/

Stored train, validation, and test partitions.

src/data/

Data loading, filtering, normalization, tokenization, and preprocessing modules.

src/models/

VarMotif-TransNet and related model implementations.

src/explainability/

Integrated Gradients, attention rollout, attribution alignment, and explanation utilities.

src/training/

Training loops, losses, optimization, checkpointing, and scheduling.

src/evaluation/

Evaluation metrics and analysis utilities.

src/utils/

Shared helper functions and utilities.

experiments/

Executable scripts for training, baselines, ablations, holdout studies, imbalance analysis, explainability, and perturbation experiments.

outputs/

Generated checkpoints, predictions, metrics, figures, and logs.

tests/

Software and pipeline tests.

requirements.txt

Python dependencies.

IMPLEMENTATION_NOTES.md

Additional implementation and research notes.

README.md

Project documentation.

5. Dataset

The framework is designed primarily for variants obtained from ClinVar. The analytical workflow supports Pathogenic, Likely Pathogenic, Benign, and Likely Benign clinical-significance classes.

Pathogenic = Pathogenic + Likely Pathogenic
Benign     = Benign + Likely Benign

Single nucleotide variants (SNVs)

Short insertions

Short deletions

Large structural variants and unresolved genomic variants should be excluded before sequence modeling.

6. Required Genomic Resources

ClinVar

Variant coordinates

REF/ALT alleles

Clinical significance

Review status

Variant identifiers

GRCh38

Reference-allele validation

Sequence-window extraction

Reference haplotype generation

GENCODE GRCh38

Gene annotation

Coding-region identification

Splice-region annotation

Non-coding context annotation

7. Data Preprocessing

ClinVar Records
↓
Target Clinical Significance Filtering
↓
Conflict Removal
↓
Review Status Filtering
↓
SNV / Short-Indel Selection
↓
Multi-Allelic Decomposition
↓
Variant Normalization
↓
GRCh38 REF Validation
↓
Duplicate Removal
↓
Binary Label Mapping

Each normalized variant should have a unique identifier of the form:

chromosome-position-reference-alternate

8. Reference and Alternate Haplotype Construction

For each variant, approximately ±1000 bp of sequence context is extracted from GRCh38. Two allele-specific genomic sequences are constructed: S_ref and S_alt. S_ref contains the original reference allele, whereas S_alt is constructed by applying the normalized REF-to-ALT transformation.

For insertions and deletions, cropping or padding is applied to maintain consistent sequence length across samples.

Variant type
REF length
ALT length
ΔL = |ALT| - |REF|
Variant-position indicator

9. k-mer Tokenization

VariantExplainAI uses overlapping k-mers with k = 5 and stride = 1. The theoretical canonical nucleotide vocabulary contains 4^5 = 1024 possible 5-mers, with additional reserved tokens available for ambiguous or padded positions.

Reference and alternate sequences use the same tokenizer and shared embedding layer.

10. Allele-Aware Representation

For each token position, the model generates E_ref and E_alt and explicitly computes:

D = E_alt - E_ref

These representations are combined with positional encoding, variant-position information, variant type, REF length, ALT length, and allele-length difference, then projected into the model hidden dimension.

11. VarMotif-TransNet

11.1 Motif-Aware 1D-CNN

Conv1D (kernel = 5, filters = 128)
↓
BatchNorm
↓
ReLU
↓
Conv1D (kernel = 3, filters = 256)
↓
BatchNorm
↓
ReLU

Same-length padding preserves genomic positional correspondence.

11.2 Sparse Transformer Encoder

Parameter

Setting

Transformer layers

4

Attention heads

8

Hidden dimension

256

Feed-forward dimension

512

Local radius

64 tokens

Global stride

32 tokens

Dropout

0.20

12. Pathogenicity Classification

CLS Representation
↓
Linear Projection
↓
Softmax
↓
Benign Probability / Pathogenic Probability

Pathogenic is treated as the positive class. The default classification threshold is 0.50.

13. Training Objective

The primary classification objective uses class-weighted cross-entropy. The complete loss is:

L_total = L_cls + λ_align × L_align
λ_align = 0.20

L_cls is the pathogenicity classification loss and L_align is the explanation-alignment loss.

14. Explainability

Integrated Gradients

Integrated Gradients is the primary attribution method and produces position-wise importance scores a = [a1, a2, ..., am] representing the contribution of genomic token positions to the predicted class.

Attention Rollout

Transformer attention matrices are recursively aggregated to estimate the influence of genomic positions on the final contextual representation. Attention rollout is intended as a complementary explanation rather than a replacement for Integrated Gradients.

15. Explanation Alignment

A variant-centered target mask is constructed around the normalized variant locus. The attribution distribution is normalized and compared with the target mask using a cosine-distance alignment objective. This regularization encourages the model to use biologically plausible variant-centered evidence while retaining the primary classification objective.

16. Training Configuration

Parameter

Setting

Reference context

±1000 bp

k-mer size

5

k-mer stride

1

Embedding dimension

128

Hidden dimension

256

CNN filters

128, 256

CNN kernels

5, 3

Transformer layers

4

Attention heads

8

FFN dimension

512

Local attention radius

64

Global attention stride

32

Dropout

0.20

Optimizer

AdamW

Initial learning rate

1e-4

Minimum learning rate

1e-6

Weight decay

1e-4

Batch size

64

Maximum epochs

80

Early-stopping patience

10

Gradient clipping

1.0

Alignment-loss weight

0.20

Random seed

42

17. Installation

Clone the repository:

git clone <repository-url>
 cd VariantExplainAI

Create a virtual environment:

python -m venv venv

Activate on Linux/macOS:

source venv/bin/activate

Activate on Windows:

venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

18. Main Dependencies

Python

PyTorch

NumPy

Pandas

scikit-learn

SciPy

Matplotlib

PyYAML

Captum

A CUDA-enabled NVIDIA GPU is strongly recommended for full-scale training.

19. Configuration

Experiment parameters are maintained in configs/config.yaml. Users can modify dataset locations and experimental settings without changing the main source code.

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

20. Dataset Preparation

Place the required input files under data/raw/ using the following organization:

data/raw/clinvar/
 data/raw/grch38/
 data/raw/gencode/

Run the preprocessing workflow before training. The preprocessing module performs ClinVar filtering, variant normalization, GRCh38 validation, REF/ALT haplotype construction, k-mer preparation, metadata encoding, and dataset partition generation.

Processed data are stored under data/processed/ and partition information under data/splits/.

21. Training VariantExplainAI

Run the primary model using:

python experiments/train_main.py

Dataset loading
       ↓
 Allele-aware representation
       ↓
 VarMotif-TransNet forward propagation
       ↓
 Classification loss
       ↓
 Explanation alignment
       ↓
 Backpropagation
       ↓
 Validation AUROC monitoring
       ↓
 Best-checkpoint selection

22. Model Outputs

Generated files are stored under outputs/:

outputs/checkpoints/
 outputs/predictions/
 outputs/metrics/
 outputs/figures/
 outputs/logs/

The main prediction output contains fields such as:

variant_id
 true_label
 predicted_label
 p_benign
 p_pathogenic
 variant_type
 gene
 region

23. Performance Metrics

Accuracy

Precision

Recall

Specificity

F1-score

AUROC

AUPRC

Confusion Matrix

ROC Curve

Precision-Recall Curve

Threshold-dependent metrics use a default classification threshold of 0.50. AUROC and AUPRC are calculated from continuous pathogenicity probabilities.

24. Baseline Experiments

Run:

python experiments/train_baselines.py

- Random Forest

- SVM-RBF

- CNN-only

- Transformer-only

- Sparse Transformer-only

The neural baselines use the same allele-aware sequence representation and stored data partitions whenever appropriate.

25. Ablation Experiments

Run:

python experiments/run_ablation.py

- Alternate allele representation

- Allele-difference encoding

- Variant metadata

- CNN motif extractor

- Sparse transformer

- Sparse versus dense attention

- Explanation alignment

- Variant-global connectivity

- Strided-global connectivity

This allows individual contributions of the proposed architecture to be quantified.

26. Gene-Wise Generalization

Run:

python experiments/run_gene_holdout.py

Under this protocol, variants belonging to the same gene are restricted to a single partition. This evaluation helps determine whether the model generalizes to unseen genes rather than learning gene-specific sequence signatures.

27. Temporal Generalization

Run:

python experiments/run_temporal_holdout.py

Earlier ClinVar data are used for model development, while newly introduced variants from a later release are reserved for testing. This protocol approximates prospective pathogenicity prediction.

28. Class-Imbalance Experiments

Run:

python experiments/run_imbalance.py

Pathogenic training ratios
1:1
1:2
1:4

Only the training partition should be modified. Validation and test distributions remain unchanged.

29. Explainability Analysis

Run:

python experiments/run_explainability.py

- Integrated Gradients

- Attention Rollout

- Variant-centered attribution analysis

- Biological annotation overlap

- Top-k functional hit analysis

- Attribution stability

30. Perturbation Faithfulness

Run:

python experiments/run_perturbation.py

- Top-attribution masking

- Variant-centered masking

- Sequence-matched control masking

- Random distal masking

- REF/ALT perturbation

ΔP = P(original) - P(perturbed)

Larger confidence drops indicate greater influence of the perturbed genomic region.

31. Statistical Testing

Perturbation results can be compared using the Wilcoxon signed-rank test with Holm multiple-comparison correction. Confidence intervals can also be generated for perturbation-based effect estimates.

32. Variant Subgroup Evaluation

SNVs

Short indels

Coding variants

Splice-region variants

Non-coding variants

This permits assessment of model behavior across different biological variant contexts.

33. Comparison with Established Predictors

The framework can support comparisons with external pathogenicity scores such as:

CADD

REVEL

ClinPred

BayesDel

PrimateAI

EVE

AlphaMissense

SpliceAI

Comparisons should only be performed on variants supported by the corresponding external predictor. For each method, evaluation should therefore use the common eligible subset:

VariantExplainAI predictions ∩ External predictor coverage

This avoids unfair comparisons involving unsupported variant classes.

34. Reproducibility

Fixed random seed

Stored dataset partitions

Configuration files

Best-model checkpoints

Saved predictions

Experiment logs

Independent evaluation scripts

Default seed: 42. The same stored partitions should be used for the main model, baselines, ablations, explainability studies, and robustness experiments.

35. Synthetic Smoke Testing

The repository includes a lightweight synthetic-data pathway for checking the software installation without requiring the complete ClinVar dataset.

Tokenization

Dataset loading

Tensor dimensions

Model forward propagation

Loss computation

Training

Checkpoint saving

Metric calculation

Explainability execution

Synthetic testing is intended only for software validation and not for reporting scientific performance.

36. Recommended Full Experimental Workflow

Obtain ClinVar, GRCh38 and GENCODE resources

Preprocess ClinVar records

Normalize and validate variants

Construct REF/ALT haplotypes

Generate dataset splits

Train VariantExplainAI

Evaluate independent test performance

Train internal baselines

Perform ablation analysis

Perform gene-wise holdout

Perform temporal holdout

Conduct class-imbalance experiments

Generate Integrated Gradients explanations

Generate attention-rollout explanations

Perform perturbation-faithfulness testing

Conduct subgroup analysis

Compare against external pathogenicity predictors

Generate publication tables and figures

37. Hardware

The intended full-scale implementation is suitable for CUDA-enabled GPU execution. The research configuration was designed around a GPU-class workstation comparable to an NVIDIA RTX 3090 with 24 GB GPU memory. Smaller configurations and synthetic experiments can be executed using lower-memory GPUs or CPU environments by reducing batch size and sequence/model dimensions.

38. Important Scientific Note

The objective of the repository is to implement the VariantExplainAI methodology reproducibly. Reported performance values from a research manuscript should not be assumed to be reproduced automatically.

ClinVar release

Filtering results

Variant normalization

Reference-genome consistency

Training environment

External predictor coverage

Randomness and deterministic-operation support

Users should generate experimental results from their own processed dataset and retained test predictions.

39. Implementation Philosophy

A major design principle of VariantExplainAI is:

The model should learn the effect of the specific REF-to-ALT transformation rather than merely memorizing genomic loci.

Therefore, careful validation of allele normalization, reference consistency, alternate-haplotype construction, and dataset leakage should be completed before interpreting final model performance.

40. Citation

If you use this implementation in academic work, please cite the corresponding VariantExplainAI research article.

@article{variantExplainAI,
   title   = {VariantExplainAI: An Explainability-Driven k-Mer CNN-Sparse Transformer Framework for ClinVar-Based Pathogenicity Prediction},
   author  = {Authors},
   journal = {Journal},
   year    = {Year}
 }

Replace the bibliographic information after publication.

41. License

Add the appropriate project license before public release. Common choices for academic research software include MIT License, Apache License 2.0, and BSD 3-Clause License. The selected license should be included in a separate LICENSE file.

42. Disclaimer

VariantExplainAI is intended for research and experimental use. It is not a certified clinical diagnostic system and should not be used independently for clinical decision-making, patient diagnosis, genetic counseling, or treatment selection.

Genomic variant classifications generated by this software should be interpreted together with validated clinical, genetic, functional, and expert-curated evidence.
