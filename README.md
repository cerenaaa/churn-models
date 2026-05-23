# Customer Churn Models

End-to-end churn prediction pipeline: feature engineering, model training (logistic regression, XGBoost, survival analysis), SHAP explainability, and deployment-ready inference.

## Structure

```
churn-models/
├── models/
│   ├── logistic_churn.py     # Baseline logistic regression + calibration
│   ├── xgboost_churn.py      # XGBoost with Optuna hyperparameter tuning
│   └── survival_churn.py     # Cox PH + Weibull AFT (time-to-churn)
├── features/
│   ├── feature_engineering.py # RFM, behavioral, contractual features
│   └── feature_store.py       # Feature versioning + retrieval
├── explainability/
│   └── shap_explainer.py      # SHAP values, force plots, waterfall charts
├── data/
│   └── synthetic_churn.py     # Synthetic dataset generator for testing
├── train.py                   # Main training script
└── requirements.txt
```

## Quickstart

```bash
pip install -r requirements.txt

# Generate synthetic data and train all models
python train.py --model all --output results/

# Train specific model
python train.py --model xgboost

# Explain predictions
python explainability/shap_explainer.py --model xgboost --sample 100
```

## Models

| Model | AUC (typical) | Use Case |
|---|---|---|
| Logistic Regression | 0.78–0.82 | Interpretable baseline |
| XGBoost | 0.84–0.90 | Production prediction |
| Cox PH | N/A | Time-to-churn, survival curves |
| Weibull AFT | N/A | Parametric survival, intervention timing |
