"""
src/train.py – LightGBM pipeline s přísným zamezením data leakage.

Pravidla:
  - Do X smí vstoupit POUZE sloupce začínající prefixem 'preprocessed_'.
  - 'preprocessed_target' je label (y) – okamžitě odstraněn z features.
  - Časový split: train ≤ 2024-12-31, test (OOS) ≥ 2025-01-01.
  - Žádný sloupec bez prefixu preprocessed_ nesmí vstoupit do modelu.

Spuštění (z kořene projektu):
    python src/train.py
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.metrics import (
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
)

INPUT_PATH  = "data/eurusd_clean.parquet"
MODEL_PATH  = "models/lgbm_clean_v1.pkl"

TRAIN_END  = pd.Timestamp("2024-12-31 23:59:59", tz="UTC")
TEST_START = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")


def main() -> None:
    print("Načítám data...")
    df = pd.read_parquet(INPUT_PATH)
    print(f"Načteno {len(df):,} řádků. Rozsah: {df.index.min()} → {df.index.max()}")

    # ── Striktní výběr features ────────────────────────────────────────────────
    TARGET_COL = "preprocessed_target"

    feat_cols = [
        c for c in df.columns
        if c.startswith("preprocessed_") and c != TARGET_COL
    ]

    # BEZPEČNOSTNÍ KONTROLA: žádný sloupec bez prefixu nesmí proniknout
    for col in feat_cols:
        if not col.startswith("preprocessed_"):
            raise ValueError(
                f"CHYBA data leakage: feature '{col}' nemá prefix 'preprocessed_'!"
            )

    if TARGET_COL not in df.columns:
        raise KeyError(f"Sloupec '{TARGET_COL}' nebyl nalezen v datech.")

    print(f"\nFeatures ({len(feat_cols)} sloupců):")
    for col in feat_cols:
        print(f"  {col}")

    X = df[feat_cols].copy()
    y = df[TARGET_COL].astype(int)

    # ── Časový split (bez shuffle – zachování kauzality) ──────────────────────
    train_mask = df.index <= TRAIN_END
    test_mask  = df.index >= TEST_START

    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_test,  y_test  = X.loc[test_mask],  y.loc[test_mask]

    print(f"\nTrain: {len(X_train):,} řádků  ({df.index[train_mask].min()} → {df.index[train_mask].max()})")
    print(f"Test:  {len(X_test):,} řádků  ({df.index[test_mask].min()} → {df.index[test_mask].max()})")
    print(f"\nDistribuce targetu (train): {y_train.value_counts().to_dict()}")
    print(f"Distribuce targetu (test):  {y_test.value_counts().to_dict()}")

    # ── Model ──────────────────────────────────────────────────────────────────
    model = lgb.LGBMClassifier(
        n_estimators  = 1000,
        learning_rate = 0.01,
        max_depth     = 6,
        num_leaves    = 31,
        random_state  = 42,
        n_jobs        = -1,
        verbose       = -1,
    )

    print("\nTrénink modelu (early stopping = 50 rund)...")
    model.fit(
        X_train,
        y_train,
        eval_set  = [(X_test, y_test)],
        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=100),
        ],
    )

    best_iter = model.best_iteration_
    print(f"\nNejlepší iterace (early stopping): {best_iter}")

    # ── Predikce na OOS datech ────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # ── Metriky ────────────────────────────────────────────────────────────────
    auc        = roc_auc_score(y_test, y_prob)
    prec_long  = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    rec_long   = recall_score   (y_test, y_pred, pos_label=1, zero_division=0)
    prec_short = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
    rec_short  = recall_score   (y_test, y_pred, pos_label=0, zero_division=0)

    # ── Výpis tabulky ──────────────────────────────────────────────────────────
    print("\n" + "=" * 56)
    print("  OOS VÝSLEDKY  (Test: 2025-01-01 → konec dat)")
    print("=" * 56)
    print(f"  {'Signál':<14}  {'Precision':>10}  {'Recall':>10}  {'AUC':>10}")
    print("-" * 56)
    print(f"  {'Long  (1)':<14}  {prec_long:>10.4f}  {rec_long:>10.4f}  {auc:>10.4f}")
    print(f"  {'Short (0)':<14}  {prec_short:>10.4f}  {rec_short:>10.4f}  {'—':>10}")
    print("=" * 56)

    print("\nPodrobná klasifikační zpráva:")
    print(classification_report(
        y_test, y_pred,
        target_names=["Short (0)", "Long  (1)"],
        digits=4,
    ))

    # ── Feature importance (top 10) ────────────────────────────────────────────
    importance = pd.Series(
        model.feature_importances_,
        index=feat_cols,
    ).sort_values(ascending=False)
    print("Top 10 features (gain importance):")
    print(importance.head(10).to_string())

    # ── Uložení modelu ─────────────────────────────────────────────────────────
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel uložen do '{MODEL_PATH}'.")


if __name__ == "__main__":
    main()
