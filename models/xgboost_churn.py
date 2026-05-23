"""
XGBoost churn model with Optuna hyperparameter tuning and calibration.
"""
import json
import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, brier_score_loss
)
from sklearn.pipeline import Pipeline
import joblib

from features.feature_engineering import build_preprocessor, prepare_data

optuna.logging.set_verbosity(optuna.logging.WARNING)


class XGBoostChurnModel:
    def __init__(self, n_trials: int = 50, cv_folds: int = 5, random_state: int = 42):
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.pipeline = None
        self.best_params = None

    def _objective(self, trial, X, y):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 5.0),
            "random_state": self.random_state,
            "eval_metric": "auc",
        }
        preprocessor = build_preprocessor()
        model = xgb.XGBClassifier(**params, use_label_encoder=False)
        pipe = Pipeline([("prep", preprocessor), ("clf", model)])
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        return scores.mean()

    def tune(self, X: pd.DataFrame, y: pd.Series):
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda t: self._objective(t, X, y), n_trials=self.n_trials)
        self.best_params = study.best_params
        print(f"Best AUC: {study.best_value:.4f}")
        print(f"Best params: {json.dumps(self.best_params, indent=2)}")
        return self

    def fit(self, X: pd.DataFrame, y: pd.Series):
        params = self.best_params or {
            "n_estimators": 300, "max_depth": 5, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8, "random_state": self.random_state,
        }
        preprocessor = build_preprocessor()
        base_model = xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric="auc")
        calibrated = CalibratedClassifierCV(base_model, method="isotonic", cv=3)
        self.pipeline = Pipeline([("prep", preprocessor), ("clf", calibrated)])
        self.pipeline.fit(X, y)
        return self

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        proba = self.pipeline.predict_proba(X)[:, 1]
        preds = (proba >= 0.5).astype(int)
        metrics = {
            "roc_auc": roc_auc_score(y, proba),
            "avg_precision": average_precision_score(y, proba),
            "brier_score": brier_score_loss(y, proba),
            "classification_report": classification_report(y, preds, output_dict=True),
        }
        print(f"ROC-AUC:  {metrics['roc_auc']:.4f}")
        print(f"Avg Prec: {metrics['avg_precision']:.4f}")
        print(f"Brier:    {metrics['brier_score']:.4f}")
        return metrics

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X)[:, 1]

    def save(self, path: str):
        joblib.dump(self.pipeline, path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "XGBoostChurnModel":
        instance = cls()
        instance.pipeline = joblib.load(path)
        return instance
