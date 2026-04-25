"""
src/features.py

Feature engineering, prioritizace a příprava X/y pro trénink.

Prioritní skupiny signálů (EURUSD / FX páry):
  1. Mean-Reversion & Extrémy – dist_ema_atr, range compression, frac_zscore
  2. Lov likvidity              – lvl_swept_*, lvl_liq_grab_*
  3. Orderflow / Mikro         – f30_ofi_z200, m1_vol_burst, m1_close_pos
  4. MTF potvrzení             – f30_mtf_H1/H4 pos_in_range, trend_alignment
  5. Volatilita & volume       – atr14, eng_vol_rank_*, atr_compression
  6. Engagement (síla pohybu) – eng_ret_atr_*, eng_dir_consistency_*
  7. Cenové úrovně             – lvl_prev_day_*, lvl_cur_day_*
"""

import pandas as pd
import numpy as np
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Šumivé / rizikové skupiny sloupců – absolutní ceny způsobují overfitting
# ---------------------------------------------------------------------------
NOISY_COLUMNS = [
    "open", "high", "low", "close",  # absolutní ceny – model dostává ATR verze
    "m1_m1_total_vol",               # absolutní objem bez kontextu
]

# ---------------------------------------------------------------------------
# EURUSD PRIORITNÍ skupiny features
# ---------------------------------------------------------------------------

# Skupina 1: Mean-Reversion & Extrémy
# EURUSD je silně mean-reverting – vzdálenost od EMA je klíčový signál
MEAN_REVERSION_FEATURES = [
    "f30_dist_ema9_atr",    # přetažení od krátkodobého EMA
    "f30_dist_ema21_atr",   # přetažení od střednědobého EMA
    "f30_dist_ema50_atr",   # vzdálenost od dlouhodobého EMA
    "f30_close_vs_ema9",
    "f30_close_vs_ema21",
    "lvl_range_compression_50",   # komprese range – předchází výbuchu / reverzi
    "lvl_range_compression_200",
    "f30_frac_zscore_50",   # z-score pozice close
    "f30_pct_off_high_20",  # vzdálenost od 20-bar high
    "f30_pct_off_low_20",   # vzdálenost od 20-bar low
    "f30_nov_pct_off_high_20_atr",
    "f30_nov_pct_off_low_20_atr",
    "f30_close_pos",
    "f30_frac_above_sma_5",
]

# Skupina 2: Lov likvidity (Liquidity Grabs)
# EURUSD respektuje session highs/lows a PDH/PDL – sweepy jsou silné signály reverze
LIQUIDITY_FEATURES = [
    "lvl_swept_prev_day_high", "lvl_swept_prev_day_low",
    "lvl_swept_cur_day_high", "lvl_swept_cur_day_low",
    "lvl_swept_cur_sess_high", "lvl_swept_cur_sess_low",
    "lvl_swept_hi_50", "lvl_swept_lo_50",
    "lvl_swept_hi_200", "lvl_swept_lo_200",
    "lvl_swept_hi_500", "lvl_swept_lo_500",
    "lvl_liq_grab_up_3", "lvl_liq_grab_dn_3",
    "lvl_liq_grab_up_5", "lvl_liq_grab_dn_5",
]

# Skupina 3: Orderflow & Mikro-dynamika
ORDERFLOW_FEATURES = [
    "f30_ofi_z200",               # order flow imbalance (hlavní signál)
    "m1_m1_vol_burst",            # náhlý nárůst objemu na M1
    "m1_m1_close_pos_in_range",   # kde zavřela M1 svíčka v range
    "m1_m1_up_ratio",
    "m1_m1_up_minus_dn",
    "m1_m1_last_sign",
    "m1_m1_last3_sign_sum",
    "m1_m1_last3_body_atr",
    "m1_m1_first_drive_atr",
    "m1_m1_close_drive_atr",
    "m1_m1_spread_burst",
    "m1_m1_dist_from_high_atr",
    "m1_m1_dist_from_low_atr",
]

# Skupina 4: Multi-timeframe potvrzení
MTF_FEATURES = [
    "f30_mtf_H1_pos_in_range",   # kde je cena v rámci H1 svíčky
    "f30_mtf_H1_ret",            # H1 return
    "f30_mtf_H1_body",
    "f30_mtf_H4_pos_in_range",   # H4 kontext (hlavní bias)
    "f30_mtf_M15_pos_in_range",
    "f30_mtf_M15_ret",
    "f30_mtf_M15_body",
    "f30_mtf_trend_alignment",   # zarovnání trendu přes TF
    "f30_m5_vs_h1_close",
]

# Skupina 5: Volatilita & volume
VOLATILITY_FEATURES = [
    "atr14",
    "eng_vol_rank_200", "eng_vol_rank_500", "eng_vol_rank_1000",
    "eng_vol_ratio_50", "eng_vol_ratio_200",
    "eng_vol_change_50",
    "eng_vol_slope_50",
    "lvl_atr_compression_50", "lvl_atr_compression_200", "lvl_atr_compression_500",
    "f30_atrn_pos_20", "f30_atrn_body", "f30_atrn_ret_10",
]

# Skupina 6: Engagement (síla a konzistence pohybu)
ENGAGEMENT_FEATURES = [
    "eng_ret_atr_3", "eng_ret_atr_5", "eng_ret_atr_10",
    "eng_ret_atr_20", "eng_ret_atr_50",
    "eng_accel_3_5", "eng_accel_5_10", "eng_accel_10_20",
    "eng_body_sum_atr_3", "eng_body_sum_atr_5", "eng_body_sum_atr_10",
    "eng_dir_consistency_10", "eng_dir_consistency_20", "eng_dir_consistency_50",
    "eng_up_count_5", "eng_dn_count_5",
    "eng_up_count_10", "eng_dn_count_10",
    "eng_consec_up_bars", "eng_consec_dn_bars",
    "eng_efficiency_10", "eng_efficiency_20", "eng_efficiency_50",
    "f30_ix_adx14__x__bb_pos",
    "f30_ix_dist_ema21_atr__x__adx14",
    "f30_frac_volret_roll50_k3", "f30_frac_volret_roll50_k5",
    "f30_nov_clv_ma5",
]

# Skupina 7: Cenové úrovně (vzdálenosti normalizované ATR)
LEVELS_FEATURES = [
    "lvl_prev_day_high_dist_atr", "lvl_prev_day_low_dist_atr", "lvl_prev_day_close_dist_atr",
    "lvl_cur_day_high_dist_atr", "lvl_cur_day_low_dist_atr",
    "lvl_cur_sess_high_dist_atr", "lvl_cur_sess_low_dist_atr",
    "lvl_hi_50_dist_atr", "lvl_lo_50_dist_atr",
    "lvl_hi_200_dist_atr", "lvl_lo_200_dist_atr",
    "lvl_hi_500_dist_atr", "lvl_lo_500_dist_atr",
    "lvl_hi_1000_dist_atr", "lvl_lo_1000_dist_atr",
    "eng_dist_high_50_atr", "eng_dist_low_50_atr",
    "eng_dist_high_200_atr", "eng_dist_low_200_atr",
    "m1_m1_sum_body_atr", "m1_m1_max_body_atr", "m1_m1_max_range_atr",
    "m1_m1_max_spread",
]

# Celkový seznam prioritních features (v pořadí skupin)
PRIORITY_FEATURES: List[str] = (
    MEAN_REVERSION_FEATURES
    + LIQUIDITY_FEATURES
    + ORDERFLOW_FEATURES
    + MTF_FEATURES
    + VOLATILITY_FEATURES
    + ENGAGEMENT_FEATURES
    + LEVELS_FEATURES
)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Přidá odvozené features na základě existujících sloupců.
    Všechny operace jsou zpětně bezpečné (používají pouze t nebo starší).
    """
    df = df.copy()
    atr = df["atr14"].replace(0, np.nan) if "atr14" in df.columns else None

    # --- ATR-normalizované rozpětí svíčky ---
    if {"high", "low"}.issubset(df.columns) and atr is not None:
        df["fe_candle_range_atr"] = (df["high"] - df["low"]) / atr

    # --- Relativní pozice close v rámci HL range ---
    if {"high", "low", "close"}.issubset(df.columns):
        hl = (df["high"] - df["low"]).replace(0, np.nan)
        df["fe_close_pos_hl"] = (df["close"] - df["low"]) / hl

    # --- Body size normalizovaný ATR ---
    if {"open", "close"}.issubset(df.columns) and atr is not None:
        df["fe_body_atr"] = (df["close"] - df["open"]) / atr

    # --- Časové features (cyklické zakódování) ---
    if hasattr(df.index, "hour"):
        hour = df.index.hour
        df["fe_hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["fe_hour_cos"] = np.cos(2 * np.pi * hour / 24)
        dow = df.index.dayofweek
        df["fe_dow_sin"] = np.sin(2 * np.pi * dow / 5)
        df["fe_dow_cos"] = np.cos(2 * np.pi * dow / 5)

    # --- Momentum: vzdálenost close od minulých close (ATR-normalized) ---
    if "close" in df.columns and atr is not None:
        for lag in [1, 3, 5]:
            df[f"fe_ret_lag{lag}_atr"] = (df["close"] - df["close"].shift(lag)) / atr

    # --- Poměr aktuálního ATR k 50-period MA ATR (komprese volatility) ---
    if atr is not None:
        df["fe_atr_ratio_50"] = atr / atr.rolling(50, min_periods=10).mean()

    # --- Interakce: OFI × MTF trend alignment (silný kombinovaný signál) ---
    if {"f30_ofi_z200", "f30_mtf_trend_alignment"}.issubset(df.columns):
        df["fe_ofi_x_trend"] = df["f30_ofi_z200"] * df["f30_mtf_trend_alignment"]

    # --- Interakce: H1 ret × H4 pos (potvrzení trendu na vyšším TF) ---
    if {"f30_mtf_H1_ret", "f30_mtf_H4_pos_in_range"}.issubset(df.columns):
        df["fe_h1ret_x_h4pos"] = df["f30_mtf_H1_ret"] * df["f30_mtf_H4_pos_in_range"]

    # --- Session features (EURUSD likviditní seance – UTC čas) ---
    # London:      07:00–12:00 UTC  (nejvyšší volume, trendy)
    # NY/London:   13:00–17:00 UTC  (overlap – nejvyšší volatilita)
    # NY afternoon:17:00–21:00 UTC
    # Dead zone:   21:00–07:00 UTC  (Asie – nízký volume, chop)
    if hasattr(df.index, "hour"):
        hour = df.index.hour
        df["fe_is_london"]    = ((hour >= 7)  & (hour < 13)).astype(int)
        df["fe_is_ny_overlap"]= ((hour >= 13) & (hour < 17)).astype(int)
        df["fe_is_dead"]      = ((hour >= 21) | (hour < 7)).astype(int)
        # Ordinální session ID (0=dead, 1=london, 2=overlap, 3=ny_afternoon)
        sess = pd.Series(0, index=df.index, dtype=int)
        sess[(hour >= 7)  & (hour < 13)] = 1
        sess[(hour >= 13) & (hour < 17)] = 2
        sess[(hour >= 17) & (hour < 21)] = 3
        df["fe_session_id"] = sess

        # Vzdálenost od London open (první bar ≥ 07:00 UTC) – ATR normalizovaná
        if "close" in df.columns and atr is not None:
            day_key = df.index.floor("D")  # midnight UTC jako klíč pro groupby
            # London open: první dostupný close v pásmu 07:00–09:00 UTC každý den
            lon_open = (
                df["close"]
                .where((hour >= 7) & (hour < 9))
                .groupby(day_key)
                .transform("first")
                .ffill()
            )
            df["fe_dist_london_open_atr"] = (df["close"] - lon_open) / atr

            # NY open: první dostupný close v pásmu 13:00–15:00 UTC každý den
            ny_open = (
                df["close"]
                .where((hour >= 13) & (hour < 15))
                .groupby(day_key)
                .transform("first")
                .ffill()
            )
            df["fe_dist_ny_open_atr"] = (df["close"] - ny_open) / atr

    return df


def select_priority_features(
    df: pd.DataFrame,
    variance_threshold: float = 1e-6,
    nan_threshold: float = 0.3,
) -> List[str]:
    """
    Vrátí průnik PRIORITY_FEATURES s dostupnými sloupci v df,
    plus engineered features (fe_*), po odfiltrování NaN/variance.

    Tato funkce preferuje robustní, ověřené signály před celým seznamem sloupců.
    """
    # Engineered features přidané funkcí add_engineered_features
    engineered = [c for c in df.columns if c.startswith("fe_")]

    candidates = [c for c in PRIORITY_FEATURES if c in df.columns] + engineered

    # Odfiltruj NaN a nulovou varianci (stejná logika jako select_features)
    selected = []
    for col in candidates:
        if df[col].isna().mean() > nan_threshold:
            continue
        if df[col].var() < variance_threshold:
            continue
        selected.append(col)

    # Deduplikace při zachování pořadí
    seen = set()
    result = []
    for c in selected:
        if c not in seen:
            seen.add(c)
            result.append(c)

    print(f"[features] Priority feature set: {len(result)} sloupců "
          f"(z {len(PRIORITY_FEATURES)} definovaných + {len(engineered)} engineered).")
    return result


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
