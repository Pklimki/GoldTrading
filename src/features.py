"""
src/features.py

Feature engineering, selekce a příprava X/y pro trénink.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Přidá odvozené features na základě existujících sloupců.
    Všechny operace jsou zpětně bezpečné (používají pouze t nebo starší).
    """
    df = df.copy()

    # --- ATR-normalizované rozpětí svíčky ---
    if {"high", "low", "atr14"}.issubset(df.columns):
        df["fe_candle_range_atr"] = (df["high"] - df["low"]) / df["atr14"].replace(0, np.nan)

    # --- Relativní pozice close v rámci HL range ---
    if {"high", "low", "close"}.issubset(df.columns):
        hl = (df["high"] - df["low"]).replace(0, np.nan)
        df["fe_close_pos_hl"] = (df["close"] - df["low"]) / hl

    # --- Body size (absolutní) normalizovaný ATR ---
    if {"open", "close", "atr14"}.issubset(df.columns):
        df["fe_body_atr"] = (df["close"] - df["open"]) / df["atr14"].replace(0, np.nan)

    # --- Časové features (cyklické zakódování) ---
    if "hour" in df.index.names or hasattr(df.index, "hour"):
        hour = df.index.hour
        df["fe_hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["fe_hour_cos"] = np.cos(2 * np.pi * hour / 24)
    elif "hour" in df.columns:
        df["fe_hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["fe_hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    if hasattr(df.index, "dayofweek"):
        dow = df.index.dayofweek
        df["fe_dow_sin"] = np.sin(2 * np.pi * dow / 5)
        df["fe_dow_cos"] = np.cos(2 * np.pi * dow / 5)
    elif "dow" in df.columns:
        df["fe_dow_sin"] = np.sin(2 * np.pi * df["dow"] / 5)
        df["fe_dow_cos"] = np.cos(2 * np.pi * df["dow"] / 5)

    # --- Momentum: vzdálenost close od minulých close (ATR-normalized) ---
    if {"close", "atr14"}.issubset(df.columns):
        for lag in [1, 3, 5]:
            df[f"fe_ret_lag{lag}_atr"] = (
                df["close"] - df["close"].shift(lag)
            ) / df["atr14"].replace(0, np.nan)

    # --- Klouzavé průměry volatility ---
    if "atr14" in df.columns:
        df["fe_atr_ratio_50"] = df["atr14"] / df["atr14"].rolling(50, min_periods=10).mean()

    return df


def select_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    variance_threshold: float = 1e-6,
    nan_threshold: float = 0.3,
) -> List[str]:
    """
    Odfiltruje features s příliš mnoha NaN nebo nulovou variancí.

    variance_threshold: min variance pro zachování sloupce
    nan_threshold:      max podíl NaN pro zachování sloupce (default 30 %)
    """
    selected = []
    dropped_nan = []
    dropped_var = []

    for col in feature_cols:
        if col not in df.columns:
            continue
        nan_ratio = df[col].isna().mean()
        if nan_ratio > nan_threshold:
            dropped_nan.append(col)
            continue
        var = df[col].var()
        if var < variance_threshold:
            dropped_var.append(col)
            continue
        selected.append(col)

    if dropped_nan:
        print(f"[features] Odstraněno {len(dropped_nan)} sloupců kvůli NaN > {nan_threshold:.0%}")
    if dropped_var:
        print(f"[features] Odstraněno {len(dropped_var)} sloupců kvůli nízké varianci")
    print(f"[features] Použito {len(selected)} features.")
    return selected


def prepare_Xy(
    df: pd.DataFrame,
    feature_cols: List[str],
    fill_na_strategy: str = "median",
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Vrátí X (features DataFrame) a y (target Series).
    NaN ve features se vyplní medianem/nulou.

    fill_na_strategy: 'median' | 'zero'
    """
    X = df[feature_cols].copy()
    y = df["target"].copy()

    if fill_na_strategy == "median":
        X = X.fillna(X.median())
    elif fill_na_strategy == "zero":
        X = X.fillna(0)
    else:
        raise ValueError(f"Neznámá strategie: {fill_na_strategy}")

    # Nahraď případné inf hodnoty NaN → pak median
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())

    return X, y
