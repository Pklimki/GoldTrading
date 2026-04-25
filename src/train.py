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


def evaluate_dual_threshold(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    long_threshold: float = 0.52,
    short_threshold: float = 0.48,
    label: str = "Test",
) -> Dict:
    """
    Duální threshold evaluace pro symetrické Long/Short predikce.

    Logika:
      - Long  signal: P(Long) >= long_threshold  (default 0.52)
      - Short signal: P(Long) <= short_threshold (default 0.48, tj. P(Short) >= 0.52)
      - Neutrální:    short_threshold < P(Long) < long_threshold → žádný vstup

    Vypisuje statistiky zvlášť pro Long a Short signály + celkové pokrytí.
    """
    proba = model.predict_proba(X)[:, 1]
    y_arr = y.to_numpy()

    long_mask = proba >= long_threshold
    short_mask = proba <= short_threshold
    neutral_mask = ~long_mask & ~short_mask

    total = len(y_arr)
    n_long = long_mask.sum()
    n_short = short_mask.sum()
    n_neutral = neutral_mask.sum()
    coverage = (n_long + n_short) / total

    print(f"\n{'='*55}")
    print(f"  {label} – Duální threshold evaluace")
    print(f"  Long  thr ≥ {long_threshold}  |  Short thr ≤ {short_threshold}")
    print(f"{'='*55}")
    print(f"  Celkem bar: {total:,}  |  Pokrytí: {coverage:.1%} "
          f"(Long: {n_long:,}, Short: {n_short:,}, Neutrál: {n_neutral:,})")

    results: Dict = {"label": label, "coverage": coverage}

    # --- Long signály ---
    if n_long > 0:
        y_long_true = y_arr[long_mask]
        prec_long = y_long_true.mean()  # precision: podíl správných Long predikcí
        print(f"\n  [LONG  ≥ {long_threshold}]  n={n_long:,}")
        print(f"    Precision (správných vstupů): {prec_long:.4f}  "
              f"({'✓' if prec_long > 0.5 else '✗'} > 50 %)")
        results["long_n"] = int(n_long)
        results["long_precision"] = float(prec_long)
    else:
        print(f"\n  [LONG  ≥ {long_threshold}]  Žádné signály – threshold příliš vysoký?")
        results["long_n"] = 0
        results["long_precision"] = float("nan")

    # --- Short signály ---
    if n_short > 0:
        y_short_true = 1 - y_arr[short_mask]  # Short je správný, když target == 0
        prec_short = y_short_true.mean()
        print(f"\n  [SHORT ≤ {short_threshold}]  n={n_short:,}")
        print(f"    Precision (správných vstupů): {prec_short:.4f}  "
              f"({'✓' if prec_short > 0.5 else '✗'} > 50 %)")
        results["short_n"] = int(n_short)
        results["short_precision"] = float(prec_short)
    else:
        print(f"\n  [SHORT ≤ {short_threshold}]  Žádné signály – threshold příliš nízký?")
        results["short_n"] = 0
        results["short_precision"] = float("nan")

    # --- Celkové AUC (bez threshold) ---
    try:
        auc = roc_auc_score(y_arr, proba)
        print(f"\n  ROC-AUC (celkové): {auc:.4f}")
        results["auc"] = float(auc)
    except Exception:
        pass

    # --- Distribuce predikcí (klíč pro ladění thresholdů) ---
    s = pd.Series(proba)
    pcts = s.describe(percentiles=[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    print(f"\n  Distribuce P(Long):")
    print(f"    p01={pcts['1%']:.3f}  p05={pcts['5%']:.3f}  p10={pcts['10%']:.3f}  "
          f"p25={pcts['25%']:.3f}  median={pcts['50%']:.3f}")
    print(f"    p75={pcts['75%']:.3f}  p90={pcts['90%']:.3f}  p95={pcts['95%']:.3f}  "
          f"p99={pcts['99%']:.3f}  max={pcts['max']:.3f}")

    print()
    return results
