"""
SHAP explainability for churn models.
Global feature importance, individual force plots, and cohort analysis.
"""
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib
from pathlib import Path


class ChurnSHAPExplainer:
    def __init__(self, model_pipeline, feature_names: list[str]):
        self.pipeline = model_pipeline
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None

    def fit(self, X_background: pd.DataFrame, n_background: int = 200):
        """Fit SHAP explainer on a background dataset."""
        preprocessor = self.pipeline.named_steps["prep"]
        X_transformed = preprocessor.transform(X_background.sample(min(n_background, len(X_background)), random_state=42))
        clf = self.pipeline.named_steps["clf"]
        # Use TreeExplainer if available, else KernelExplainer
        try:
            base_clf = clf.calibrated_classifiers_[0].estimator
            self.explainer = shap.TreeExplainer(base_clf)
        except Exception:
            self.explainer = shap.KernelExplainer(clf.predict_proba, X_transformed)
        return self

    def explain(self, X: pd.DataFrame) -> np.ndarray:
        preprocessor = self.pipeline.named_steps["prep"]
        X_transformed = preprocessor.transform(X)
        self.shap_values = self.explainer.shap_values(X_transformed)
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[1]
        return self.shap_values

    def plot_summary(self, X: pd.DataFrame, max_display: int = 20, save_path: str = None):
        """Global feature importance (beeswarm plot)."""
        if self.shap_values is None:
            self.explain(X)
        preprocessor = self.pipeline.named_steps["prep"]
        X_transformed = preprocessor.transform(X)
        shap.summary_plot(self.shap_values, X_transformed, max_display=max_display, show=False)
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_force(self, X: pd.DataFrame, idx: int = 0, save_path: str = None):
        """Force plot for a single prediction."""
        if self.shap_values is None:
            self.explain(X)
        preprocessor = self.pipeline.named_steps["prep"]
        X_transformed = preprocessor.transform(X)
        shap.initjs()
        plot = shap.force_plot(
            self.explainer.expected_value if not isinstance(self.explainer.expected_value, list)
            else self.explainer.expected_value[1],
            self.shap_values[idx],
            X_transformed[idx],
            matplotlib=True,
            show=False
        )
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def top_drivers(self, X: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        """Returns mean absolute SHAP values (global importance ranking)."""
        if self.shap_values is None:
            self.explain(X)
        preprocessor = self.pipeline.named_steps["prep"]
        try:
            feature_names = preprocessor.get_feature_names_out()
        except Exception:
            feature_names = [f"f{i}" for i in range(self.shap_values.shape[1])]
        importance = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": np.abs(self.shap_values).mean(axis=0)
        }).sort_values("mean_abs_shap", ascending=False).head(n)
        return importance

    def individual_explanation(self, X: pd.DataFrame, idx: int) -> pd.DataFrame:
        """Human-readable explanation for a single customer."""
        if self.shap_values is None:
            self.explain(X)
        preprocessor = self.pipeline.named_steps["prep"]
        try:
            feature_names = preprocessor.get_feature_names_out()
        except Exception:
            feature_names = [f"f{i}" for i in range(self.shap_values.shape[1])]
        explanation = pd.DataFrame({
            "feature": feature_names,
            "shap_value": self.shap_values[idx]
        }).sort_values("shap_value", key=abs, ascending=False)
        explanation["direction"] = explanation["shap_value"].apply(lambda x: "↑ Increases churn" if x > 0 else "↓ Decreases churn")
        return explanation.head(10)
