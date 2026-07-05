# Bank Customer Churn Prediction — KNN

Predicts whether a bank customer will churn (exit) using a K-Nearest Neighbors (KNN) classifier.

## Dataset
Bank Customer Churn dataset (included in this repo as `Churn_Modelling.csv`). Features: customer demographics, account details, activity metrics. Target: `Exited` (1 = churned, 0 = stayed).

## Approach
- Preprocessing: encoding categorical features, feature scaling
- Model: K-Nearest Neighbors (KNN) classifier
- Train/test split, evaluated on 2000 test samples
- Tool: Google Colab

## Results

**Accuracy: 85.35%**

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 0 (Retained) | 0.86 | 0.98 | 0.91 | 1607 |
| 1 (Churned)  | 0.80 | 0.34 | 0.48 | 393  |

**Confusion Matrix**

|              | Predicted: No | Predicted: Yes |
|--------------|---------------|-----------------|
| **Actual: No**  | 1574 | 33  |
| **Actual: Yes** | 260  | 133 |

### Notes
- Model performs well on the majority class (retained customers) but struggles with recall 
  on churned customers (34%) — likely due to class imbalance (1607 vs 393).
- Future improvement: try class weighting, SMOTE, or tuning the `k` value to boost minority-class recall.

## How to Run
1. Open the `.ipynb` file in Google Colab or Jupyter
2. Install dependencies: `pip install pandas scikit-learn matplotlib seaborn`
3. Run cells sequentially

## Tech Stack
Python, scikit-learn, pandas, matplotlib/seaborn
