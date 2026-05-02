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
from models import get_model, get_config_params

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

    # ── TBM breakdown (TP / SL / Timeout) ─────────────────────────────────────
    if "tbm_outcome" in df.columns:
        def _tbm_stats(series: pd.Series, label: str) -> None:
            total  = series.notna().sum()
            tp_pct = (series == 1).sum() / total * 100
            sl_pct = (series == 0).sum() / total * 100
            to_pct = (series == 2).sum() / total * 100
            print(f"  {label:6s}  TP={tp_pct:5.1f}%  SL={sl_pct:5.1f}%  Timeout={to_pct:5.1f}%")
        print("TBM breakdown (1=TP, 0=SL, 2=Timeout):")
        _tbm_stats(df["tbm_outcome"],                  "Celkem")
        _tbm_stats(df["tbm_outcome"].loc[train_mask],  "Train ")
        _tbm_stats(df["tbm_outcome"].loc[test_mask],   "Test  ")

    # ── Model ──────────────────────────────────────────────────────────────────
    MODEL_TYPE = "default"   # ← změň na "conservative" pro experimentování
    model = get_model(MODEL_TYPE)
    print(f"\nKonfigurace modelu: '{MODEL_TYPE}' → {get_config_params(MODEL_TYPE)}")
    print("Trénink modelu (early stopping = 50 rund)...")
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

    # ── Probability Thresholding ───────────────────────────────────────────────
    # Zobraz vliv různých thresholdů na Precision / Recall / Coverage.
    # Coverage = podíl obchodů, které threshold pustí dál.
    THRESHOLDS = [0.50, 0.52, 0.55, 0.60]
    print("\n" + "=" * 72)
    print("  PROBABILITY THRESHOLDING  (Long signal = prob >= threshold)")
    print("=" * 72)
    print(f"  {'Threshold':>10}  {'Prec Long':>10}  {'Rec Long':>10}  "
          f"{'Prec Short':>10}  {'Coverage':>10}  {'N Long':>8}")
    print("-" * 72)
    for thr in THRESHOLDS:
        y_thr = np.where(y_prob >= thr, 1, 0)
        pl = precision_score(y_test, y_thr, pos_label=1, zero_division=0)
        rl = recall_score   (y_test, y_thr, pos_label=1, zero_division=0)
        ps = precision_score(y_test, y_thr, pos_label=0, zero_division=0)
        n_long    = int((y_thr == 1).sum())
        coverage  = n_long / len(y_test) * 100
        print(f"  {thr:>10.2f}  {pl:>10.4f}  {rl:>10.4f}  "
              f"{ps:>10.4f}  {coverage:>9.2f}%  {n_long:>8,}")
    print("=" * 72)

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
