"""
Synthetic churn dataset generator.
Creates a realistic customer dataset with behavioral, contractual, and support features.
"""
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


def generate_churn_dataset(n_customers: int = 5000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)

    df = pd.DataFrame()
    df["customer_id"] = [f"CUST_{i:05d}" for i in range(n_customers)]
    df["tenure_months"] = np.random.exponential(scale=24, size=n_customers).clip(1, 120).astype(int)
    df["monthly_charges"] = np.random.normal(65, 20, n_customers).clip(20, 150).round(2)
    df["total_charges"] = (df["tenure_months"] * df["monthly_charges"] * np.random.uniform(0.85, 1.0, n_customers)).round(2)
    df["contract_type"] = np.random.choice(["month-to-month", "one-year", "two-year"], n_customers, p=[0.55, 0.25, 0.20])
    df["payment_method"] = np.random.choice(["credit_card", "bank_transfer", "electronic_check", "mailed_check"], n_customers)
    df["num_products"] = np.random.choice([1, 2, 3, 4, 5], n_customers, p=[0.20, 0.35, 0.25, 0.12, 0.08])
    df["support_calls_6m"] = np.random.poisson(1.5, n_customers)
    df["avg_monthly_usage"] = np.random.normal(500, 150, n_customers).clip(50, 1000).round(1)
    df["usage_trend"] = np.random.normal(0, 0.1, n_customers)  # positive = growing usage
    df["days_since_last_login"] = np.random.exponential(scale=15, size=n_customers).clip(0, 180).astype(int)
    df["nps_score"] = np.random.choice(range(0, 11), n_customers, p=[0.05, 0.03, 0.04, 0.05, 0.07, 0.10, 0.12, 0.15, 0.18, 0.12, 0.09])
    df["has_paperless_billing"] = np.random.binomial(1, 0.6, n_customers)
    df["num_dependents"] = np.random.poisson(0.8, n_customers).clip(0, 5)

    # Churn probability model (ground truth)
    churn_logit = (
        -2.0
        + 1.5 * (df["contract_type"] == "month-to-month").astype(float)
        - 0.5 * (df["contract_type"] == "two-year").astype(float)
        + 0.03 * df["support_calls_6m"]
        - 0.02 * df["tenure_months"]
        + 0.01 * df["monthly_charges"]
        - 0.3 * df["nps_score"] / 10
        + 2.0 * df["usage_trend"].clip(-0.5, 0)
        + 0.005 * df["days_since_last_login"]
        + 0.3 * (df["payment_method"] == "electronic_check").astype(float)
        - 0.2 * df["num_products"]
        + np.random.normal(0, 0.3, n_customers)
    )
    df["churn_prob"] = 1 / (1 + np.exp(-churn_logit))
    df["churned"] = (np.random.uniform(size=n_customers) < df["churn_prob"]).astype(int)
    df["months_to_churn"] = np.where(
        df["churned"] == 1,
        np.random.exponential(scale=6, size=n_customers).clip(1, df["tenure_months"]).astype(int),
        np.nan
    )

    print(f"Generated {n_customers} customers | Churn rate: {df['churned'].mean():.1%}")
    return df.drop(columns=["churn_prob"])


if __name__ == "__main__":
    df = generate_churn_dataset()
    df.to_csv("data/churn_data.csv", index=False)
    print(df.describe())
