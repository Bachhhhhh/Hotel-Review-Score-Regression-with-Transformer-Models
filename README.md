# Hotel Review Score Regression with Transformer Models

---

## Table of Contents

- [Project Overview](#-project-overview)
- [Dataset Description](#-dataset-description)
- [Model Description](#-model-description)
  - [RobertaRegressor (roberta-base)](#robertainregressor-roberta-base)
  - [TransformerRegressor (bert-base-cased)](#transformerregressor-bert-base-cased)
- [Requirements](#-requirements)
- [Workflow](#-workflow)
- [Performance Comparison](#-performance-comparison)

---

## Project Overview

This project tackles a **text regression** task: given a hotel review written in English, the model predicts a satisfaction score normalized to the range **[0, 1]** (originally on a 0–10 scale).

Two pretrained Transformer models are fine-tuned and compared:
- **`FacebookAI/roberta-base`** — with a lightweight 2-layer MLP regression head
- **`google-bert/bert-base-cased`** — tested with both the simple head and a deeper MLP head `[100 → 50 → 20 → 10 → 1]`

The training objective is to minimize **Mean Squared Error (MSE)** between predicted and true scores. Models are evaluated using **MSE**, **MAE**, and **RMSE** on a held-out test set.

---

## Dataset Description

| Property | Details |
|---|---|
| **Source** | Google Drive (`data.csv`) |
| **Size** | ~3.98 MB |
| **Columns** | `normalized_content` (text), `score` (float, 0–10) |
| **Language** | English |
| **Task** | Regression |

**Sample data:**

| normalized_content | score |
|---|---|
| Very friendly staff. Nice welcome. | 8.0 |
| It was a superior experience. Accommodation was... | 10.0 |
| The staff! They were amazing and so friendly... | 10.0 |

**Preprocessing:**
- Scores are normalized to `[0, 1]` by dividing by 10.
- Data is split **60 / 20 / 20** into Train / Validation / Test sets using stratified random splitting (`random_state=42`).

---

## Model Description

### RobertaRegressor (`roberta-base`)

The baseline model uses `FacebookAI/roberta-base` as the encoder. The `[CLS]`-equivalent token (`<s>`) representation is passed through a 2-layer MLP regression head.

```
Input Text
    │
[RoBERTa Encoder] (frozen pre-trained weights, fine-tuned)
    │
CLS vector (hidden_size = 768)
    │
Dropout(0.2) → Linear(768, 128) → ReLU
    │
Dropout(0.2) → Linear(128, 1)
    │
Sigmoid → ŷ ∈ [0, 1]
```

**Hyperparameters:**

| Parameter | Value |
|---|---|
| Pretrained model | `FacebookAI/roberta-base` |
| Max sequence length | 128 |
| Batch size | 48 |
| Optimizer | AdamW |
| Learning rate | 2e-5 |
| Epochs | 5 |
| Hidden dim (MLP) | 128 |
| Dropout | 0.2 |

---

### TransformerRegressor (`bert-base-cased`)

This variant replaces the encoder with `google-bert/bert-base-cased` and uses a **deeper MLP head** with layers `[100 → 50 → 20 → 10 → 1]` to explore the effect of head depth on regression performance.

```
Input Text
    │
[BERT Encoder] (bert-base-cased)
    │
CLS vector (hidden_size = 768)
    │
Dropout → Linear(768, 100) → ReLU
    │
Dropout → Linear(100, 50) → ReLU
    │
Dropout → Linear(50, 20) → ReLU
    │
Dropout → Linear(20, 10) → ReLU
    │
Linear(10, 1)
    │
Sigmoid → ŷ ∈ [0, 1]
```

**Hyperparameters:**

| Parameter | Value |
|---|---|
| Pretrained model | `google-bert/bert-base-cased` |
| Max sequence length | 256 |
| Batch size | 48 |
| Optimizer | AdamW |
| Learning rate | 2e-5 |
| Epochs | 5 |
| Head architecture | 768 → 100 → 50 → 20 → 10 → 1 |
| Dropout | 0.2 |

---

## Requirements

```bash
pip install transformers torch scikit-learn pandas tqdm
```

Or install from a requirements file:

```bash
pip install -r requirements.txt
```

**`requirements.txt`:**
```
torch>=2.0.0
transformers>=4.30.0
scikit-learn>=1.2.0
pandas>=1.5.0
numpy>=1.23.0
tqdm>=4.65.0
gdown>=4.7.0
```

---

## Workflow

```
1. Environment Setup
   └── Install dependencies (transformers, torch, ...)

2. Data Download & Loading
   └── Download data.csv via gdown
   └── Load with pandas, extract `normalized_content` and `score`

3. Data Preprocessing
   └── Normalize scores: score / 10.0, clipped to [0, 1]
   └── Train / Val / Test split: 60% / 20% / 20%

4. Tokenization & Dataset
   └── Wrap data in TextRegressionDataset (PyTorch Dataset)
   └── Tokenize with AutoTokenizer (max_length = 128 or 256)
   └── Create DataLoader (batch_size = 48, pin_memory = True)

5. Model Initialization
   └── Load pretrained encoder (RoBERTa or BERT)
   └── Attach regression head (MLP + Sigmoid)
   └── Move model to DEVICE

6. Training Loop (5 Epochs)
   └── Forward pass → MSE Loss → Backprop → AdamW update
   └── Evaluate on validation set each epoch
   └── Save best model checkpoint (lowest Val MSE)

7. Evaluation on Test Set
   └── Load best checkpoint
   └── Compute MSE, MAE, RMSE on test set

8. Model Comparison
   └── Compare RobertaRegressor vs TransformerRegressor
```

---

## Performance Comparison

Results on the **test set** after 5 epochs of fine-tuning:

| Model | Encoder | Head Architecture | Max Length | MSE | MAE | RMSE |
|---|---|---|---|---|---|---|
| **RobertaBaseRegressor** | `roberta-base` | 768 → 128 → 1 | 128 | 0.0102 | 0.0679 | 0.1011 |
| **BertBaseCasedRegressor** | `bert-base-cased` | 768 → 128 → 1 | 256 | 0.0111 | 0.0696 | 0.1054 |
| **BertBaseCasedv2Regressor** | `bert-base-cased` | 768 → 100 → 50 → 20 → 10 → 1 | 256 | 0.0221 | 0.1076 | 0.1486 |

### Conclusion

- **RobertaBaseRegressor** achieves the best performance across all metrics (MSE: 0.0102, MAE: 0.0679, RMSE: 0.1011), demonstrating that RoBERTa's training procedure — which removes the Next Sentence Prediction objective and uses dynamic masking — yields stronger text representations for regression tasks.

- **BertBaseCasedRegressor** with the same shallow head but using `bert-base-cased` scores slightly lower, suggesting that the choice of pretrained encoder has a greater impact on performance than increasing the input sequence length from 128 to 256.

- **BertBaseCasedv2Regressor** uses a significantly deeper MLP head `[100 → 50 → 20 → 10 → 1]` but performs the worst of the three. Stacking multiple FC layers with ReLU activations makes gradient optimization harder and may cause the model to overfit or converge to a suboptimal solution on this dataset.

- A lightweight 2-layer MLP head is sufficient for this regression task. The quality of the pretrained encoder matters more than the depth of the regression head. RoBERTa-base with a simple head delivers the best trade-off between model complexity and predictive accuracy.

---
