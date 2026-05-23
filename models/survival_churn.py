"""
Survival analysis for churn: Cox Proportional Hazards + Weibull AFT.
Estimates time-to-churn and intervention windows.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, WeibullAFTFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DURATION_COL = "tenure_months"
EVENT_COL = "churned"

SURVIVAL_FEATURES = [
    "monthly_charges", "num_products", "support_calls_6m",
    "avg_monthly_usage", "usage_trend", "nps_score",
    "has_paperless_billing", "is_month_to_month",
    "charge_per_product", "support_intensity", "low_nps",
]


class SurvivalChurnModel:
    def __init__(self):
        self.cox = CoxPHFitter(penalizer=0.1)
        self.aft = WeibullAFTFitter(penalizer=0.1)
        self.scaler = StandardScaler()
        self._feature_cols = None

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        from features.feature_engineering import build_rfm_features, build_risk_features
        df = build_rfm_features(df)
        df = build_risk_features(df)
        df = df.dropna(subset=[DURATION_COL, EVENT_COL])
        df = df[df[DURATION_COL] > 0]
        return df

    def fit(self, df: pd.DataFrame):
        df = self._prepare(df)
        available = [f for f in SURVIVAL_FEATURES if f in df.columns]
        self._feature_cols = available

        survival_df = df[[DURATION_COL, EVENT_COL] + available].copy()
        scaled = self.scaler.fit_transform(survival_df[available])
        survival_df[available] = scaled

        print("Fitting Cox PH model...")
        self.cox.fit(survival_df, duration_col=DURATION_COL, event_col=EVENT_COL)
        self.cox.print_summary(decimals=3)

        print("\nFitting Weibull AFT model...")
        self.aft.fit(survival_df, duration_col=DURATION_COL, event_col=EVENT_COL)
        self.aft.print_summary(decimals=3)
        return self

    def predict_median_survival(self, df: pd.DataFrame) -> pd.Series:
        """Predict median months-to-churn for each customer."""
        df = self._prepare(df)
        X = df[self._feature_cols].copy()
        X_scaled = pd.DataFrame(self.scaler.transform(X), columns=self._feature_cols)
        return self.aft.predict_median(X_scaled)

    def predict_churn_probability_at(self, df: pd.DataFrame, t: int = 12) -> pd.Series:
        """Probability of churning within t months."""
        df = self._prepare(df)
        X = df[self._feature_cols].copy()
        X_scaled = pd.DataFrame(self.scaler.transform(X), columns=self._feature_cols)
        survival_funcs = self.cox.predict_survival_function(X_scaled)
        # P(churn by t) = 1 - S(t)
        t_actual = min(t, survival_funcs.index.max())
        return 1 - survival_funcs.loc[t_actual]

    def plot_survival_curves(self, df: pd.DataFrame, segment_col: str = "contract_type", save_path: str = None):
        """Kaplan-Meier curves by segment."""
        df = self._prepare(df)
        kmf = KaplanMeierFitter()
        fig, ax = plt.subplots(figsize=(10, 6))
        for segment in df[segment_col].unique():
            mask = df[segment_col] == segment
            kmf.fit(df.loc[mask, DURATION_COL], df.loc[mask, EVENT_COL], label=segment)
            kmf.plot_survival_function(ax=ax)
        ax.set_title("Kaplan-Meier Survival Curves by Segment")
        ax.set_xlabel("Months")
        ax.set_ylabel("Survival Probability (not churned)")
        ax.legend()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig
