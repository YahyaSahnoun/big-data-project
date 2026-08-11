# Unbiased Threshold Moving Report

## Methodology (No Data Leakage)

To eliminate the data leakage from the previous experiment, the dataset is now split into **three** separate sets:

| Set | Size | Purpose |
|-----|------|---------|
| **Train** | 60% (120,000 rows) | Model fitting only |
| **Validation** | 20% (40,000 rows) | Threshold selection (maximize F1) |
| **Test** | 20% (40,000 rows) | Final unbiased evaluation |

- **Dataset**: 3,172,296 total rows × 66 columns → subsampled to 200,000 rows (stratified) for tractability.
- **Class imbalance**: `scale_pos_weight = 22.51` (heavily imbalanced — ~4.25% positive class).
- **No oversampling/undersampling** was applied. Imbalance handled via `scale_pos_weight`.
- The **threshold was selected on the Validation set** and then applied to the **unseen Test set** — no leakage.

## Results

| Metric | Default (T=0.50) | Optimal (T=0.67) |
|--------|:-----------------:|:-----------------:|
| **F1-Score** | 0.1519 | **0.2130** |
| **Precision** | — | 0.1532 |
| **Recall** | — | 0.3492 |

> **Key Finding**: Threshold moving from 0.50 → 0.67 improved the **unbiased F1-score by +40%** (0.1519 → 0.2130) on the completely held-out test set.

> **Note**: The overall F1-score is low due to the extreme class imbalance (22.5:1 ratio). This is expected — the model is trading precision for recall. Further improvements could come from:
> - Training on the full 3.17M dataset (vs 200k subsample)
> - SMOTE or hybrid resampling
> - More estimators / deeper trees
> - Feature engineering or selection

## Visual Representation

![Unbiased Threshold Moving Analysis](C:/Users/Mr Tarouzi/.gemini/antigravity-ide/brain/56bd35fb-740c-4776-9d7e-931e86caa699/unbiased_threshold_metrics.png)
