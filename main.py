"""
main.py

Spouštěcí skript – celá pipeline od načtení dat po uložení modelu.

Použití:
    python main.py --data data/eurusd.parquet --model lgbm
    python main.py --data data/eurusd.parquet --train-end 2023-12-31 --test-start 2024-01-01
    python main.py --data data/xauusd.parquet --long-threshold 0.54 --short-threshold 0.46
"""

import argparse
import sys
from pathlib import Path

# Přidej src/ do cesty, aby fungovaly importy
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import load_and_prepare
from src.features import add_engineered_features, select_priority_features, prepare_Xy
from src.model import build_lgbm, build_xgb, save_model
from src.train import (
    compute_class_weight_ratio,
    fit_model,
    evaluate,
    evaluate_dual_threshold,
    find_optimal_threshold,
    print_feature_importance,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="XAUUSD / EURUSD ML Predictor")
    parser.add_argument("--data", type=str, default="data/xauusd.parquet", help="Cesta k datovému souboru (.parquet nebo .csv)")
    parser.add_argument("--model", choices=["lgbm", "xgb"], default="lgbm", help="Typ modelu")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Podíl dat pro trénink (0-1) – ignorováno, pokud jsou zadána data data")
    parser.add_argument("--train-end", type=str, default="2023-12-31",
                        help="Poslední datum tren. sady vrčetně (default: 2023-12-31)")
    parser.add_argument("--test-start", type=str, default="2024-01-01",
                        help="První datum test. sady (default: 2024-01-01, OOS)")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Podíl z trénovacích dat pro validaci / early stopping")
    parser.add_argument("--long-threshold", type=float, default=0.52,
                        help="Threshold pro Long signál: P(Long) >= X (default: 0.52)")
    parser.add_argument("--short-threshold", type=float, default=0.48,
                        help="Threshold pro Short signál: P(Long) <= X (default: 0.48)")
    parser.add_argument("--output", type=str, default=None, help="Cesta pro uložení modelu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # -----------------------------------------------------------------------
    # 1. Načtení a příprava dat
    # -----------------------------------------------------------------------
    print("\n[1/5] Načítání dat...")
    print(f"      Train: do {args.train_end} | Test (OOS): od {args.test_start}")
    train_df, test_df, feature_cols = load_and_prepare(
        filepath=args.data,
        train_end=args.train_end,
        test_start=args.test_start,
    )

    # -----------------------------------------------------------------------
    # 2. Feature engineering
    # -----------------------------------------------------------------------
    print("\n[2/5] Feature engineering...")
    train_df = add_engineered_features(train_df)
    test_df = add_engineered_features(test_df)

    # Použij prioritní feature set zaměřený na MTF a Orderflow
    feature_cols = select_priority_features(train_df)

    # -----------------------------------------------------------------------
    # 3. Rozděl train → train + validace (pro early stopping / threshold tuning)
    # -----------------------------------------------------------------------
    val_cut = int(len(train_df) * (1 - args.val_ratio))
    val_df = train_df.iloc[val_cut:]
    train_df = train_df.iloc[:val_cut]
    print(f"\n[3/5] Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    X_train, y_train = prepare_Xy(train_df, feature_cols)
    X_val, y_val = prepare_Xy(val_df, feature_cols)
    X_test, y_test = prepare_Xy(test_df, feature_cols)

    # -----------------------------------------------------------------------
    # 4. Sestavení a trénink modelu
    # -----------------------------------------------------------------------
    print(f"\n[4/5] Trénink modelu ({args.model.upper()})...")
    ratio = compute_class_weight_ratio(y_train)

    if args.model == "lgbm":
        model = build_lgbm(class_weight_ratio=ratio)
    else:
        model = build_xgb(class_weight_ratio=ratio)

    model = fit_model(model, X_train, y_train, X_val=X_val, y_val=y_val)

    # -----------------------------------------------------------------------
    # 5. Evaluace
    # -----------------------------------------------------------------------
    print("\n[5/5] Evaluace...")
    print(f"[main] Long threshold: ≥ {args.long_threshold}  |  Short threshold: ≤ {args.short_threshold}")

    evaluate_dual_threshold(
        model, X_val, y_val,
        long_threshold=args.long_threshold,
        short_threshold=args.short_threshold,
        label="Validace",
    )
    evaluate_dual_threshold(
        model, X_test, y_test,
        long_threshold=args.long_threshold,
        short_threshold=args.short_threshold,
        label="Test (OOS)",
    )
    print_feature_importance(model, feature_cols, top_n=20)

    # -----------------------------------------------------------------------
    # Uložení modelu
    # -----------------------------------------------------------------------
    output_path = args.output or f"models/{args.model}_model.pkl"
    save_model(model, output_path)
    print(f"\nHotovo! Model uložen do: {output_path}")
    print(f"Spuštění příkladem: python main.py --data {args.data} "
          f"--long-threshold {args.long_threshold} "
          f"--short-threshold {args.short_threshold}")


if __name__ == "__main__":
    main()
