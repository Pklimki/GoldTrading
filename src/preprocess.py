"""
src/preprocess.py – Feature engineering pro EURUSD M1 bez data leakage.

Pravidla:
  - Všechny vstupní features musí popisovat minulost (t nebo t-n).
  - Target (preprocessed_target) je budoucnost (t+1) – NIKDY do features.
  - Veškeré features mají prefix 'preprocessed_'.

Spuštění (z kořene projektu):
    python src/preprocess.py
"""

import pandas as pd
import numpy as np

INPUT_PATH  = "data/EURUSD_M1_full.parquet"
OUTPUT_PATH = "data/eurusd_clean.parquet"

ATR_PERIOD  = 14
RSI_PERIOD  = 14
EMA_SHORT   = 50
EMA_LONG    = 200
H1_EMA_LEN  = 200   # EMA pro MTF H1 trend
VOL_ZSCORE_WINDOW = 500
VOL_SURGE_WINDOW  = 20
CET_TZ      = "Europe/Berlin"   # CET (UTC+1) / CEST (UTC+2) s DST


# ── Pomocné funkce ─────────────────────────────────────────────────────────────

def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series,
                period: int = 14) -> pd.Series:
    """ATR pomocí Wilderova EMA (com = period-1)."""
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period, adjust=False).mean()


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI pomocí Wilderova EMA (com = period-1)."""
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


# ── Hlavní funkce ──────────────────────────────────────────────────────────────

def main() -> None:
    print("Načítám data...")
    df = pd.read_parquet(INPUT_PATH)

    # ── Index: unix → UTC DatetimeIndex ────────────────────────────────────────
    df.index = pd.to_datetime(df["bar_ts_utc"], unit="s", utc=True)
    df.index.name = "datetime_utc"
    df = df.sort_index()
    print(f"Načteno {len(df):,} řádků. Rozsah: {df.index.min()} → {df.index.max()}")

    open_  = df["open"]
    high   = df["high"]
    low    = df["low"]
    close  = df["close"]
    volume = df["tick_volume"]

    # ── ATR pro normalizaci ────────────────────────────────────────────────────
    atr14    = compute_atr(high, low, close, ATR_PERIOD)
    safe_atr = atr14.replace(0, np.nan)
    hl_range = (high - low).replace(0, np.nan)

    # ── Target (BUDOUCNOST t+1) ────────────────────────────────────────────────
    # POZOR: close.shift(-1) je výhradně pro label – NIKDY do features!
    next_close = close.shift(-1)
    target     = (next_close > close).astype("float64")
    target.loc[next_close.isna()] = np.nan   # poslední řádek nemá budoucnost

    # ── Výstupní DataFrame ─────────────────────────────────────────────────────
    out = pd.DataFrame(index=df.index)

    # ── Price Action (normalizováno ATR) ───────────────────────────────────────
    body         = (close - open_).abs()
    upper_shadow = high - pd.concat([open_, close], axis=1).max(axis=1)
    lower_shadow = pd.concat([open_, close], axis=1).min(axis=1) - low

    out["preprocessed_body_atr"]         = body         / safe_atr
    out["preprocessed_upper_shadow_atr"] = upper_shadow / safe_atr
    out["preprocessed_lower_shadow_atr"] = lower_shadow / safe_atr
    out["preprocessed_close_pos"]        = (close - low) / hl_range  # [0, 1]

    # ── Momentum ───────────────────────────────────────────────────────────────
    out["preprocessed_rsi14"] = compute_rsi(close, RSI_PERIOD)
    out["preprocessed_roc5"]  = close.pct_change(5)
    out["preprocessed_roc10"] = close.pct_change(10)
    out["preprocessed_roc20"] = close.pct_change(20)

    # ── Volatilita: ATR normalizovaný cenou ────────────────────────────────────
    out["preprocessed_atr14_norm"] = atr14 / close

    # ── Trend: logaritmus poměru close / EMA ──────────────────────────────────
    ema50  = close.ewm(span=EMA_SHORT, min_periods=EMA_SHORT,  adjust=False).mean()
    ema200 = close.ewm(span=EMA_LONG,  min_periods=EMA_LONG,   adjust=False).mean()
    out["preprocessed_log_close_ema50"]  = np.log(close / ema50)
    out["preprocessed_log_close_ema200"] = np.log(close / ema200)

    # ── MTF Trend: H1 EMA(200) ─────────────────────────────────────────────────
    # Algoritmus bez leakage:
    #   1. Resample M1 → H1 close (close poslední M1 svíčky v každé H1 svíčce)
    #   2. EMA(200) na H1 close
    #   3. shift(1) – pro každou M1 svíčku uvnitř H1 periody X vidíme
    #      EMA z H1 periody X-1 (plně uzavřené), nikoliv té aktuální
    #   4. ffill zpět na M1 index
    print("Počítám H1 EMA(200) pro MTF trend...")
    h1_close           = close.resample("1h").last().dropna()
    h1_ema200          = h1_close.ewm(span=H1_EMA_LEN, min_periods=H1_EMA_LEN,
                                      adjust=False).mean()
    h1_ema200_lagged   = h1_ema200.shift(1)                      # bez leakage
    h1_ema200_m1       = h1_ema200_lagged.reindex(close.index, method="ffill")
    out["preprocessed_h1_trend"] = np.log(close / h1_ema200_m1)

    # ── Volatility Regime: Z-score ATR ────────────────────────────────────────
    # Z-score ATR14 vůči průměru/std posledních 500 svíček (pouze minulost).
    atr_roll_mean = atr14.rolling(VOL_ZSCORE_WINDOW, min_periods=VOL_ZSCORE_WINDOW).mean()
    atr_roll_std  = atr14.rolling(VOL_ZSCORE_WINDOW, min_periods=VOL_ZSCORE_WINDOW).std()
    out["preprocessed_vol_zscore"] = (
        (atr14 - atr_roll_mean) / atr_roll_std.replace(0, np.nan)
    ).astype("float32")

    # ── Volume Dynamics: Volume Surge ─────────────────────────────────────────
    # Poměr aktuálního volume vůči průměru 20 předchozích svíček.
    # shift(1) zajistí, že průměr je z [t-20 … t-1], nikoli [t-19 … t].
    vol_ma20 = volume.rolling(VOL_SURGE_WINDOW, min_periods=VOL_SURGE_WINDOW).mean().shift(1)
    out["preprocessed_volume_surge"] = (
        volume / vol_ma20.replace(0, np.nan)
    ).astype("float32")

    # ── Time & Sessions ────────────────────────────────────────────────────────
    cet_idx     = df.index.tz_convert(CET_TZ)
    cet_hour    = np.asarray(cet_idx.hour,      dtype="int64")
    cet_minute  = np.asarray(cet_idx.minute,    dtype="int64")
    dow_arr     = np.asarray(cet_idx.dayofweek, dtype="int64")
    cet_min_day = cet_hour * 60 + cet_minute    # minuty od půlnoci CET

    # Seance (binární 0/1)
    london_mask = (cet_min_day >= 8 * 60)  & (cet_min_day < 17 * 60)  # 08–17 CET
    ny_mask     = (cet_min_day >= 13 * 60) & (cet_min_day < 22 * 60)  # 13–22 CET
    asia_mask   = (cet_min_day >= 0)       & (cet_min_day < 9 * 60)   # 00–09 CET

    out["preprocessed_session_london"] = london_mask.astype("float32")
    out["preprocessed_session_ny"]     = ny_mask.astype("float32")
    out["preprocessed_session_asia"]   = asia_mask.astype("float32")

    # Vzdálenost od otevření Londýnské seance [minuty, může být záporná]
    out["preprocessed_dist_london_open"] = (cet_min_day - 8 * 60).astype("float32")

    # Cyklické kódování: čas v rámci dne (minutová přesnost)
    minutes_in_day = 24 * 60
    hour_angle     = 2.0 * np.pi * cet_min_day / minutes_in_day
    out["preprocessed_hour_sin"] = np.sin(hour_angle).astype("float32")
    out["preprocessed_hour_cos"] = np.cos(hour_angle).astype("float32")

    # Cyklické kódování: den v týdnu (0=Po … 6=Ne)
    dow_angle = 2.0 * np.pi * dow_arr / 7.0
    out["preprocessed_dow_sin"] = np.sin(dow_angle).astype("float32")
    out["preprocessed_dow_cos"] = np.cos(dow_angle).astype("float32")

    # ── Target ────────────────────────────────────────────────────────────────
    out["preprocessed_target"] = target

    # ── Základní OHLC + spread pro backtest (bez prefixu) ────────────────────
    out["open"]   = open_.values
    out["high"]   = high.values
    out["low"]    = low.values
    out["close"]  = close.values
    out["spread"] = df["spread"].values   # raw spread v bodech (MT5 points)

    # ── Čistění – odstraň NaN (warmup + poslední řádek bez targetu) ───────────
    rows_before = len(out)
    out = out.dropna()
    rows_after  = len(out)
    print(f"Odstraněno {rows_before - rows_after:,} řádků s NaN (warmup + konec).")
    print(f"Výsledek: {rows_after:,} řádků.")

    # ── Ověření NaN ───────────────────────────────────────────────────────────
    nan_counts = out.isna().sum()
    if nan_counts.any():
        print("[VAROVÁNÍ] Nalezeny NaN hodnoty:")
        print(nan_counts[nan_counts > 0])
    else:
        print("Kontrola NaN: OK – žádné NaN hodnoty.")

    print(f"\nSloupce ({len(out.columns)}):")
    for col in out.columns:
        print(f"  {col}")

    # ── Uložení ────────────────────────────────────────────────────────────────
    out.to_parquet(OUTPUT_PATH)
    print(f"\nUloženo do '{OUTPUT_PATH}'.")


if __name__ == "__main__":
    main()
