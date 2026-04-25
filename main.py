"""
main.py

Spouštěcí skript – celá pipeline od načtení dat po uložení modelu.

Použití:
    python main.py --data data/xauusd.csv --model lgbm
    python main.py --data data/xauusd.csv --model xgb --train-ratio 0.75
"""

import argparse
import sys
from pathlib import Path

# Přidej src/ do cesty, aby fungovaly importy
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import load_and_prepare
from src.features import add_engineered_features, select_features, prepare_Xy
from src.model import build_lgbm, build_xgb, save_model
from src.train import (
    compute_class_weight_ratio,
    fit_model,
    evaluate,
    find_optimal_threshold,
    print_feature_importance,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="XAUUSD ML Predictor")
    parser.add_argument("--data", type=str, default="data/xauusd.csv", help="Cesta k CSV souboru")
    parser.add_argument("--model", choices=["lgbm", "xgb"], default="lgbm", help="Typ modelu")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Podíl dat pro trénink (0-1)")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Podíl z trénovacích dat pro validaci / early stopping")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Klasifikační threshold (None = auto-optimalizace)")
    parser.add_argument("--output", type=str, default=None, help="Cesta pro uložení modelu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # -----------------------------------------------------------------------
    # 1. Načtení a příprava dat
    # -----------------------------------------------------------------------
    print("\n[1/5] Načítání dat...")
    train_df, test_df, feature_cols = load_and_prepare(
        filepath=args.data,
        train_ratio=args.train_ratio,
    )

    # -----------------------------------------------------------------------
    # 2. Feature engineering
    # -----------------------------------------------------------------------
    print("\n[2/5] Feature engineering...")
    train_df = add_engineered_features(train_df)
    test_df = add_engineered_features(test_df)

    # Aktualizuj seznam features po přidání nových
    all_cols = [c for c in train_df.columns if c != "target"]
    feature_cols = select_features(train_df, all_cols)

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

    if args.threshold is None:
        threshold = find_optimal_threshold(model, X_val, y_val, metric="f1")
    else:
        threshold = args.threshold

    evaluate(model, X_val, y_val, threshold=threshold, label="Validace")
    evaluate(model, X_test, y_test, threshold=threshold, label="Test (OOS)")
    print_feature_importance(model, feature_cols, top_n=20)

    # -----------------------------------------------------------------------
    # Uložení modelu
    # -----------------------------------------------------------------------
    output_path = args.output or f"models/{args.model}_model.pkl"
    save_model(model, output_path)
    print(f"\nHotovo! Model uložen do: {output_path}")


if __name__ == "__main__":
    main()
