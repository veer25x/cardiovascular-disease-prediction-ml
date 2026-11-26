# cardiovascular-disease-prediction-ml
Comparative ML study predicting cardiovascular disease on 70K patient records using Decision Tree, Naive Bayes, KNN &amp; ANN, with CV, feature importance, and a conceptual clinical risk-screening system.
# AI-Based Cardiovascular Disease Prediction
Comparative classification of cardiovascular disease using **Decision Tree, Naive Bayes, KNN, and ANN (MLP)** on 70,000 patient records.

## Highlights
- Cleaning of implausible BP/height/weight values, feature engineering (BMI, pulse pressure), encoding, scaling inside CV pipelines (no leakage)
- Grid-search tuning + 5-fold stratified CV + 80/20 hold-out test
- Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix
- Permutation + tree-based feature importance, bias/variance curves, error analysis, paired t-tests
- Conceptual healthcare risk-screening system with a saved `predict_patient()` inference function

## Results (hold-out test)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Decision Tree | 0.728 | 0.727 | 0.719 | 0.723 | 0.786 |
| Naive Bayes | 0.708 | 0.777 | 0.574 | 0.660 | 0.778 |
| KNN | 0.721 | 0.733 | 0.686 | 0.708 | 0.777 |
| ANN (MLP) | 0.733 | 0.762 | 0.669 | 0.713 | 0.800 |

**Key findings**
- Blood pressure (`ap_hi`, `ap_lo`, pulse pressure) is the strongest predictor, followed by age and high cholesterol
- Decision Tree gives the best F1/recall; ANN gives the best accuracy/AUC, and the F1 difference is not statistically significant (p≈0.09)
- All models fall in a narrow 71–73% accuracy band, so data quality is the limiting factor

## Tech Stack
Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · Jupyter

## Run
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter joblib scipy
jupyter notebook CVD_Prediction.ipynb   # keep cardio_train.csv in the same folder
```

## Dataset
Cardiovascular Disease dataset (Kaggle, 70,000 records, 11 features, binary target `cardio`).

## Disclaimer
Educational project for decision support only; not a diagnostic tool.
