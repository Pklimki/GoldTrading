"""
src/train.py

Trénovací pipeline s evaluací a výpisem výsledků.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)


def compute_class_weight_ratio(y: pd.Series) -> float:
    """
    Vrátí scale_pos_weight = počet negativních / počet pozitivních příkladů.
    Vhodné pro XGBoost a LightGBM při nevyvážených třídách.
    """
    neg = (y == 0).sum()
    pos = (y == 1).sum()
    ratio = neg / pos if pos > 0 else 1.0
    print(f"[train] Class weight ratio (neg/pos): {ratio:.3f}  (neg={neg}, pos={pos})")
    return ratio


def fit_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
) -> Any:
    """
    Natrénuje model s volitelnou validační sadou pro early stopping.
    Podporuje LightGBM a XGBoost callbacks.
    """
    fit_kwargs: Dict = {}

    model_name = type(model).__name__

    if X_val is not None and y_val is not None:
        if "LGBM" in model_name:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["callbacks"] = [
                __import__("lightgbm").early_stopping(50, verbose=False),
                __import__("lightgbm").log_evaluation(100),
            ]
        elif "XGB" in model_name:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["verbose"] = False

    model.fit(X_train, y_train, **fit_kwargs)
    return model


def evaluate(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.5,
    label: str = "Test",
) -> Dict:
    """
    Komplexní evaluace: Accuracy, Precision, Recall, F1, AUC.
    Vrátí dict s metrikami.
    """
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)

    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds, zero_division=0)
    rec = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)
    auc = roc_auc_score(y, proba)

    print(f"\n{'='*50}")
    print(f"  {label} Evaluace  (threshold={threshold})")
    print(f"{'='*50}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}  ← 'Když vstoupím, mám pravdu?'")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-score:  {f1:.4f}")
    print(f"  ROC-AUC:   {auc:.4f}")
    print(f"\n{classification_report(y, preds, target_names=['Short (0)', 'Long (1)'])}")

    cm = confusion_matrix(y, preds)
    print(f"  Confusion matrix:\n{cm}\n")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "threshold": threshold,
        "label": label,
    }


def find_optimal_threshold(
    model: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    metric: str = "f1",
) -> float:
    """
    Projde thresholdy 0.40–0.65 a vrátí ten s nejlepší zvolenou metrikou.
    Použij na validační (ne testovací!) sadě.
    """
    proba = model.predict_proba(X_val)[:, 1]
    best_score = -1.0
    best_thr = 0.5

    for thr in np.arange(0.40, 0.66, 0.01):
        preds = (proba >= thr).astype(int)
        if metric == "f1":
            score = f1_score(y_val, preds, zero_division=0)
        elif metric == "precision":
            score = precision_score(y_val, preds, zero_division=0)
        elif metric == "accuracy":
            score = accuracy_score(y_val, preds)
        else:
            raise ValueError(f"Neznámá metrika: {metric}")

        if score > best_score:
            best_score = score
            best_thr = round(thr, 2)

    print(f"[train] Optimální threshold ({metric}): {best_thr:.2f}  →  {metric}={best_score:.4f}")
    return best_thr


def print_feature_importance(model: Any, feature_cols: List[str], top_n: int = 20) -> None:
    """
    Vypíše top-N nejdůležitějších features.
    """
    from src.model import get_feature_importance
    df_imp = get_feature_importance(model, feature_cols, top_n=top_n)
    print(f"\n{'='*50}")
    print(f"  Top-{top_n} Feature Importance")
    print(f"{'='*50}")
    print(df_imp.to_string(index=False))
