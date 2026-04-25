"""
src/data_loader.py

Načítání, čištění a time-series split XAUUSD dat.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List


# Sloupce, které mohou obsahovat data leakage (výstupy jiných modelů)
LEAKY_COLUMNS = [
    "pred_p_raw", "pred_p_iso", "pred_p_iso_v2", "pred_iso_version",
    "pred_thr_p55", "pred_thr_p56", "pred_thr_p57", "pred_thr_p60",
    "pred_thr_p55_reg", "pred_thr_p56_reg", "pred_thr_p57_reg", "pred_thr_p60_reg",
    "pred_p_short", "pred_p_med", "pred_p_long", "pred_p_lgbm", "pred_p_l2",
    "pred_p_ens_eq", "pred_p_ens_iv", "pred_p_ens_eq_iso",
    "pred_thr_p55_ens", "pred_thr_p56_ens", "pred_thr_p57_ens", "pred_thr_p60_ens",
    "label_h1",  # hotový label z budoucnosti
]

# Pomocné/identifikační sloupce, které nejsou features
META_COLUMNS = [
    "bar_ts_unix", "bar_ts", "week_key", "year", "month", "day",
    "hour", "minute", "chunk", "mode", "regime_id", "regime", "macro_regime",
]


def load_csv(filepath: str | Path) -> pd.DataFrame:
    """
    Načte CSV, převede bar_ts na datetime index a seřadí chronologicky.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Soubor nenalezen: {filepath}")

    df = pd.read_csv(filepath, low_memory=False)

    # Převod timestampu
    if "bar_ts" in df.columns:
        df["bar_ts"] = pd.to_datetime(df["bar_ts"], utc=True, errors="coerce")
        df = df.set_index("bar_ts").sort_index()
    elif "bar_ts_unix" in df.columns:
        df["bar_ts"] = pd.to_datetime(df["bar_ts_unix"], unit="s", utc=True)
        df = df.set_index("bar_ts").sort_index()
    else:
        raise ValueError("Dataset neobsahuje sloupec 'bar_ts' ani 'bar_ts_unix'.")

    print(f"[data_loader] Načteno {len(df):,} řádků | {df.index.min()} → {df.index.max()}")
    return df


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vytvoří binární target:
        target = 1 pokud Close(t+1) > Close(t), jinak 0.

    Poslední řádek nemá known budoucnost → odstraní se.
    """
    df = df.copy()
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    # Poslední svíčka nemá label – odstraníme
    df = df.iloc[:-1]
    return df


def drop_leaky_and_meta_columns(df: pd.DataFrame, extra_drop: List[str] | None = None) -> pd.DataFrame:
    """
    Odstraní sloupce s potenciálním data leakage a čistě meta sloupce.
    """
    to_drop = set(LEAKY_COLUMNS + META_COLUMNS)
    if extra_drop:
        to_drop.update(extra_drop)

    existing = [c for c in to_drop if c in df.columns]
    df = df.drop(columns=existing)
    print(f"[data_loader] Odstraněno {len(existing)} leaky/meta sloupců.")
    return df


def time_series_split(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Rozdělí data chronologicky – žádné náhodné míchání.

    train_ratio: podíl dat pro trénink (default 80 %)
    Vrací: (train_df, test_df)
    """
    n = len(df)
    split_idx = int(n * train_ratio)

    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()

    print(
        f"[data_loader] Train: {len(train):,} řádků "
        f"({train.index.min().date()} → {train.index.max().date()})"
    )
    print(
        f"[data_loader] Test:  {len(test):,} řádků  "
        f"({test.index.min().date()} → {test.index.max().date()})"
    )

    # Zkontroluj balanci tříd
    for name, split in [("Train", train), ("Test", test)]:
        counts = split["target"].value_counts(normalize=True)
        up = counts.get(1, 0)
        dn = counts.get(0, 0)
        print(f"[data_loader] {name} class balance → Long: {up:.1%}  Short: {dn:.1%}")

    return train, test


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Vrátí seznam feature sloupců (vše kromě 'target').
    """
    return [c for c in df.columns if c != "target"]


def load_and_prepare(
    filepath: str | Path,
    train_ratio: float = 0.8,
    extra_drop: List[str] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Kompletní pipeline: načti → vytvoř target → odstraň leaky → rozděl.

    Vrací: (train_df, test_df, feature_columns)
    """
    df = load_csv(filepath)
    df = create_target(df)
    df = drop_leaky_and_meta_columns(df, extra_drop=extra_drop)

    # Odstraň řádky s NaN v targetu nebo příliš mnoho NaN ve features
    before = len(df)
    df = df.dropna(subset=["target"])
    df = df.dropna(thresh=int(len(df.columns) * 0.5))  # řádky, kde >50 % sloupců je NaN
    print(f"[data_loader] Odstraněno {before - len(df)} řádků s NaN.")

    train, test = time_series_split(df, train_ratio=train_ratio)
    feature_cols = get_feature_columns(train)

    return train, test, feature_cols


if __name__ == "__main__":
    import sys

    data_path = Path("data") / "xauusd.csv"
    if len(sys.argv) > 1:
        data_path = Path(sys.argv[1])

    train, test, features = load_and_prepare(data_path)
    print(f"\nFeatures ({len(features)}): {features[:5]} ...")
