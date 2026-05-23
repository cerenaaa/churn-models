"""
Main training script: generates data, trains models, evaluates, and saves artifacts.
"""
import argparse
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

from data.synthetic_churn import generate_churn_dataset
from features.feature_engineering import prepare_data
from models.xgboost_churn import XGBoostChurnModel
from models.survival_churn import SurvivalChurnModel


def train_xgboost(df, output_dir: Path):
    print("\n=== XGBoost Churn Model ===")
    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = XGBoostChurnModel(n_trials=30)
    model.tune(X_train, y_train)
    model.fit(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)
    model.save(str(output_dir / "xgboost_churn.pkl"))
    with open(output_dir / "xgboost_metrics.json", "w") as f:
        json.dump({k: v for k, v in metrics.items() if k != "classification_report"}, f, indent=2)
    return model


def train_survival(df, output_dir: Path):
    print("\n=== Survival Analysis (Cox PH + Weibull AFT) ===")
    model = SurvivalChurnModel()
    model.fit(df)
    probs_12m = model.predict_churn_probability_at(df, t=12)
    print(f"\nMedian predicted 12-month churn probability: {probs_12m.median():.3f}")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["xgboost", "survival", "all"], default="all")
    parser.add_argument("--n_customers", type=int, default=5000)
    parser.add_argument("--output", default="results")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    print("Generating synthetic churn dataset...")
    df = generate_churn_dataset(n_customers=args.n_customers)
    df.to_csv("data/churn_data.csv", index=False)

    if args.model in ("xgboost", "all"):
        train_xgboost(df, output_dir)

    if args.model in ("survival", "all"):
        train_survival(df, output_dir)

    print(f"\n✓ Done. Artifacts saved to {output_dir}/")


if __name__ == "__main__":
    main()
