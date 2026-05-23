"""
Feature engineering for churn models.
Computes RFM features, behavioral signals, and contract risk indicators.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


NUMERIC_FEATURES = [
    "tenure_months", "monthly_charges", "total_charges",
    "num_products", "support_calls_6m", "avg_monthly_usage",
    "usage_trend", "days_since_last_login", "nps_score",
    "has_paperless_billing", "num_dependents",
]

CATEGORICAL_FEATURES = ["contract_type", "payment_method"]

TARGET = "churned"


def build_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Recency-Frequency-Monetary features for churn context."""
    df = df.copy()
    df["recency_score"] = pd.qcut(df["days_since_last_login"], q=5, labels=[5, 4, 3, 2, 1]).astype(int)
    df["frequency_score"] = pd.qcut(df["avg_monthly_usage"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    df["monetary_score"] = pd.qcut(df["monthly_charges"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    df["rfm_score"] = df["recency_score"] + df["frequency_score"] + df["monetary_score"]
    return df


def build_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineered risk signals."""
    df = df.copy()
    df["charge_per_product"] = df["monthly_charges"] / df["num_products"].clip(1)
    df["support_intensity"] = df["support_calls_6m"] / df["tenure_months"].clip(1)
    df["is_month_to_month"] = (df["contract_type"] == "month-to-month").astype(int)
    df["is_high_value"] = (df["monthly_charges"] > df["monthly_charges"].quantile(0.75)).astype(int)
    df["low_nps"] = (df["nps_score"] <= 6).astype(int)  # detractor
    df["declining_usage"] = (df["usage_trend"] < -0.05).astype(int)
    return df


def build_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline([("scaler", StandardScaler())])
    categorical_transformer = Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

    engineered_numeric = NUMERIC_FEATURES + [
        "rfm_score", "recency_score", "frequency_score", "monetary_score",
        "charge_per_product", "support_intensity", "is_month_to_month",
        "is_high_value", "low_nps", "declining_usage",
    ]

    return ColumnTransformer([
        ("num", numeric_transformer, engineered_numeric),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])


def prepare_data(df: pd.DataFrame):
    df = build_rfm_features(df)
    df = build_risk_features(df)
    X = df.drop(columns=[TARGET, "customer_id", "months_to_churn"], errors="ignore")
    y = df[TARGET]
    return X, y
