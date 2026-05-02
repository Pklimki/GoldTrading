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

INPUT_PATH    = "data/EURUSD_M1_full.parquet"
RESAMPLE_FREQ = "5min"   # cílový timeframe: "5min" = M5 (výchozí), "15min" = M15
OUTPUT_PATH   = (
    "data/eurusd_clean.parquet"
    if RESAMPLE_FREQ == "5min"
    else f"data/eurusd_clean_{RESAMPLE_FREQ.replace('min', 'm')}.parquet"
)

ATR_PERIOD  = 14
RSI_PERIOD  = 14
EMA_SHORT   = 50
EMA_LONG    = 200
H1_EMA_LEN  = 200   # EMA pro MTF H1 trend
VOL_ZSCORE_WINDOW = 500
VOL_SURGE_WINDOW  = 20
RANGE_24H_BARS    = 1440          # 1 440 minut = 24 hodin
RANGE_7D_BARS     = 10_080        # 10 080 minut = 7 dní
LAG_PERIODS       = (1, 2, 3, 5)  # lagy pro krátkou paměť
SWING_WINDOW      = 20            # okno pro detekci swing high/low
ATR_LONG_PERIOD   = 100           # dlouhodobý ATR pro volatility compression
PIP_SIZE          = 0.0001        # EURUSD: 1 pip = 0.0001
CET_TZ      = "Europe/Berlin"   # CET (UTC+1) / CEST (UTC+2) s DST

ER_WINDOW    = 10   # počet barů pro Efficiency Ratio (Kaufman ER)

# Triple Barrier Method (TBM)  –  cílový čas = 120 minut
_TBM_HORIZONS = {"5min": 24, "15min": 12, "1min": 120}
TBM_HORIZON   = _TBM_HORIZONS.get(RESAMPLE_FREQ, 24)  # barů do exitu
TBM_PT_MULT   = 3.0   # Profit Taking bariéra = ATR(14) × TBM_PT_MULT
TBM_SL_MULT   = 2.0   # Stop Loss bariéra  = ATR(14) × TBM_SL_MULT
ATR_200_PERIOD    = 200   # pro Vol Persistence feature
ATR_TRADABLE_PIPS = 1.5   # minimální ATR(14) v pipech pro is_tradable (uvolněno z 2.5)

# News Filter – používá se v backtestu (ne v tréninku).
# True = vynechá vstupy 14:30–14:45 a 16:00–16:15 CET (NFP, FOMC apod.)
FILTER_NEWS = False


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


def compute_tbm_labels(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    horizon: int = TBM_HORIZON,
    pt_arr: np.ndarray = None,  # abs. vzdálenost TP od Close per bar (price units)
    sl_arr: np.ndarray = None,  # abs. vzdálenost SL od Close per bar (price units)
) -> tuple:
    """
    Triple Barrier Method – vektorizovaná implementace (numpy stride tricks).

    Bariéry jsou dynamické (per-bar), typicky ATR-násobky:
      pt_arr = ATR(14) * TBM_PT_MULT   → TP úroveň = Close(T) + pt_arr[T]
      sl_arr = ATR(14) * TBM_SL_MULT   → SL úroveň = Close(T) - sl_arr[T]

    Pro každou svíčku T kontroluje budoucí okno T+1 … T+horizon:
      - Pokud High(T+k) >= Close(T) + pt_arr[T] první → label = 1 (TP)
      - Pokud Low(T+k)  <= Close(T) - sl_arr[T] první → label = 0 (SL)
      - Pokud žádná bariéra není zasažena              → label = 0 (Timeout)
      - Tie (stejný bar zasáhne obě)                   → label = 0 (konzervativní)

    Posledních `horizon` řádků dostane NaN (chybí budoucnost).

    Returns
    -------
    labels  : pd.Series  float64  – 1 = TP, 0 = SL/Timeout
    outcome : pd.Series  float64  – 1 = TP, 0 = SL, 2 = Timeout  (pro reporting)
    """
    pt = np.asarray(pt_arr, dtype="float64")
    sl = np.asarray(sl_arr, dtype="float64")

    n          = len(close)
    close_arr  = np.asarray(close, dtype="float64")
    high_arr   = np.asarray(high,  dtype="float64")
    low_arr    = np.asarray(low,   dtype="float64")

    # Okna budoucích high/low: shape (n - horizon, horizon)
    # win_h[i, k] = high[i + k + 1],  k = 0 … horizon-1
    win_h = np.lib.stride_tricks.sliding_window_view(high_arr, horizon + 1)[:, 1:]
    win_l = np.lib.stride_tricks.sliding_window_view(low_arr,  horizon + 1)[:, 1:]

    m        = win_h.shape[0]              # počet barů s plným oknem
    tp_level = close_arr[:m] + pt[:m]      # (m,) – per-bar TP
    sl_level = close_arr[:m] - sl[:m]      # (m,) – per-bar SL

    tp_hits = win_h >= tp_level[:, None]   # (m, horizon) bool
    sl_hits = win_l <= sl_level[:, None]

    # Index prvního zásahu (0-based v okně); horizon = "nikdy nezasaženo"
    tp_first = np.where(tp_hits.any(axis=1), tp_hits.argmax(axis=1), horizon)
    sl_first = np.where(sl_hits.any(axis=1), sl_hits.argmax(axis=1), horizon)

    # Label: 1 pouze pokud TP zasažen STRIKTNĚ před SL
    labels_arr = np.where(tp_first < sl_first, 1.0, 0.0)

    # Outcome pro reporting: 1=TP, 0=SL, 2=Timeout
    outcome_arr = np.select(
        [tp_first < sl_first,   # TP hit first
         sl_first < horizon],   # SL hit first (nebo tie)
        [1, 0],
        default=2,              # žádná bariéra → timeout
    ).astype("float64")

    # Posledních `horizon` řádků → NaN
    labels_full  = np.full(n, np.nan)
    outcome_full = np.full(n, np.nan)
    labels_full[:m]  = labels_arr
    outcome_full[:m] = outcome_arr

    return (
        pd.Series(labels_full,  index=close.index),
        pd.Series(outcome_full, index=close.index),
    )


# ── Hlavní funkce ──────────────────────────────────────────────────────────────

def main() -> None:
    print("Načítám data...")
    df = pd.read_parquet(INPUT_PATH)

    # ── Index: unix → UTC DatetimeIndex ────────────────────────────────────────
    df.index = pd.to_datetime(df["bar_ts_utc"], unit="s", utc=True)
    df.index.name = "datetime_utc"
    df = df.sort_index()
    print(f"Načteno {len(df):,} M1 řádků. Rozsah: {df.index.min()} → {df.index.max()}")

    # ── Resample M1 → M5 ──────────────────────────────────────────────────────
    print("Resampluju M1 → M5 (5minutové svíčky)...")
    df = df[["open", "high", "low", "close", "tick_volume", "spread"]].resample(RESAMPLE_FREQ).agg({
        "open":        "first",
        "high":        "max",
        "low":         "min",
        "close":       "last",
        "tick_volume": "sum",
        "spread":      "last",
    }).dropna(subset=["open", "close"])
    print(f"Po resamplu: {len(df):,} {RESAMPLE_FREQ}-barů.")

    open_  = df["open"]
    high   = df["high"]
    low    = df["low"]
    close  = df["close"]
    volume = df["tick_volume"]

    # ── ATR pro normalizaci ────────────────────────────────────────────────────
    atr14    = compute_atr(high, low, close, ATR_PERIOD)
    safe_atr = atr14.replace(0, np.nan)
    hl_range = (high - low).replace(0, np.nan)

    # ── Target: Triple Barrier Method (dynamické ATR bariéry) ─────────────────
    # POZOR: používá budoucí high/low (T+1..T+horizon) VÝHRADNĚ pro label.
    # PT = ATR(14)[T] × TBM_PT_MULT,  SL = ATR(14)[T] × TBM_SL_MULT
    print(f"Počítám TBM target (horizon={TBM_HORIZON}, PT=ATR×{TBM_PT_MULT}, SL=ATR×{TBM_SL_MULT})...")
    target, tbm_outcome = compute_tbm_labels(
        close, high, low,
        horizon = TBM_HORIZON,
        pt_arr  = (atr14 * TBM_PT_MULT).values,
        sl_arr  = (atr14 * TBM_SL_MULT).values,
    )
    # Statistika TBM (informativní výpis)
    m_valid = tbm_outcome.notna().sum()
    tp_pct  = (tbm_outcome == 1).sum() / m_valid * 100
    sl_pct  = (tbm_outcome == 0).sum() / m_valid * 100
    to_pct  = (tbm_outcome == 2).sum() / m_valid * 100
    print(f"TBM distribuce: TP={tp_pct:.1f}%  SL={sl_pct:.1f}%  Timeout={to_pct:.1f}%")

    # ── Výstupní DataFrame ─────────────────────────────────────────────────────
    out = pd.DataFrame(index=df.index)

    # ── Price Action (normalizováno ATR) ───────────────────────────────────────
    body         = (close - open_).abs()
    upper_shadow = high - pd.concat([open_, close], axis=1).max(axis=1)
    lower_shadow = pd.concat([open_, close], axis=1).min(axis=1) - low

    body_atr_raw = body / safe_atr
    out["preprocessed_body_atr"]         = body_atr_raw
    out["preprocessed_upper_shadow_atr"] = upper_shadow / safe_atr
    out["preprocessed_lower_shadow_atr"] = lower_shadow / safe_atr
    out["preprocessed_close_pos"]        = (close - low) / hl_range  # [0, 1]

    # ── Momentum ───────────────────────────────────────────────────────────────
    rsi14_raw = compute_rsi(close, RSI_PERIOD)
    out["preprocessed_rsi14"] = rsi14_raw
    out["preprocessed_roc5"]  = close.pct_change(5)
    out["preprocessed_roc10"] = close.pct_change(10)
    out["preprocessed_roc20"] = close.pct_change(20)

    # ── Volatilita: ATR normalizovaný cenou ────────────────────────────────────
    out["preprocessed_atr14_norm"] = atr14 / close

    # Volatilitní filtr: 1 pokud ATR(14) > ATR_TRADABLE_PIPS – obchodovatelné podmínky
    # Model (a backtest) budou obchodovat POUZE tyto bary.
    out["preprocessed_is_tradable"] = (
        (atr14 > ATR_TRADABLE_PIPS * PIP_SIZE).astype("float32")
    )

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
    volume_surge_raw = (volume / vol_ma20.replace(0, np.nan))
    out["preprocessed_volume_surge"] = volume_surge_raw.astype("float32")

    # ── Lagged Features (krátká paměť) ────────────────────────────────────────
    # Každý lag .shift(n) dává hodnotu uzavřené svíčky z T-n.
    # Model vidí gradient: roste RSI? Graduje volume? Zrychluje se tělo?
    print("Počítám lagged features...")
    for lag in LAG_PERIODS:
        out[f"preprocessed_body_atr_lag{lag}"]     = body_atr_raw.shift(lag).astype("float32")
        out[f"preprocessed_rsi14_lag{lag}"]        = rsi14_raw.shift(lag).astype("float32")
        out[f"preprocessed_volume_surge_lag{lag}"] = volume_surge_raw.shift(lag).astype("float32")

    # ── Rolling Trend Windows (dlouhá paměť) ──────────────────────────────────
    # Vždy shift(1): okno [T-N … T-1], nikdy T.
    print("Počítám rolling trend windows...")

    # 24h range: (max(high) - min(low)) za posledních 1440 minut
    high_24h = high.rolling(RANGE_24H_BARS, min_periods=RANGE_24H_BARS).max().shift(1)
    low_24h  = low.rolling(RANGE_24H_BARS, min_periods=RANGE_24H_BARS).min().shift(1)
    out["preprocessed_range_24h"] = (high_24h - low_24h).astype("float32")

    # 7d high-low kanál: kde je aktuální close v rámci 7denního kanálu (0–1)
    high_7d  = high.rolling(RANGE_7D_BARS, min_periods=RANGE_7D_BARS).max().shift(1)
    low_7d   = low.rolling(RANGE_7D_BARS, min_periods=RANGE_7D_BARS).min().shift(1)
    range_7d = (high_7d - low_7d).replace(0, np.nan)
    out["preprocessed_close_vs_7d_highlow"] = ((close - low_7d) / range_7d).astype("float32")

    # H1 EMA200 slope: (EMA200[T-1] - EMA200[T-4]) / 3 na H1 = sklon za 3 hodiny
    # h1_ema200_lagged je již posunutá o 1 H1 bar (viz výše) → beze extra leakage
    h1_slope_raw  = (h1_ema200_lagged - h1_ema200_lagged.shift(3)) / 3.0
    out["preprocessed_slope_ema200_h1"] = (
        h1_slope_raw.reindex(close.index, method="ffill")
    ).astype("float32")

    # ── Advanced Price Action ──────────────────────────────────────────────────
    print("Počítám advanced price action features...")

    # Swing Points: je předchozí High/Low lokálním extrémen za 20 svíček?
    # Vždy shift(1) – porovnáváme uzavřenou svíčku T-1 s oknem [T-20…T-1].
    high_shifted  = high.shift(1)
    low_shifted   = low.shift(1)
    roll_max_high = high.rolling(SWING_WINDOW, min_periods=SWING_WINDOW).max().shift(1)
    roll_min_low  = low.rolling(SWING_WINDOW, min_periods=SWING_WINDOW).min().shift(1)
    out["preprocessed_is_high_20"] = (high_shifted == roll_max_high).astype("float32")
    out["preprocessed_is_low_20"]  = (low_shifted  == roll_min_low).astype("float32")

    # Volatility Compression: poměr krátkodobého ATR vs. dlouhodobého ATR
    # < 1 = komprese (konsolidace), > 1 = expanze
    atr100 = compute_atr(high, low, close, ATR_LONG_PERIOD)
    out["preprocessed_atr_ratio"] = (
        (atr14 / atr100.replace(0, np.nan)).astype("float32")
    )

    # Price Extremes: vzdálenost Close(T-1) od 24h High v pipech
    # Záporná hodnota = close je pod 24h high (čím větší abs, tím dál)
    out["preprocessed_dist_from_24h_high"] = (
        ((close.shift(1) - high_24h) / PIP_SIZE).astype("float32")
    )

    # Momentum Climax: počet po sobě jdoucích svíček se stejným znaménkem body
    # Kladné tělo (Up) = close > open, záporné (Down) = close < open
    body_sign = np.sign(close - open_)   # +1, -1 nebo 0
    body_sign_shifted = body_sign.shift(1)

    def _consecutive_same_sign(s: pd.Series) -> pd.Series:
        """Počítá délku aktuální série shodných znaménk (zpětně)."""
        arr    = s.values
        result = np.zeros(len(arr), dtype="float32")
        count  = 0
        for i in range(len(arr)):
            if i == 0 or arr[i] == 0 or arr[i] != arr[i - 1]:
                count = 1 if arr[i] != 0 else 0
            else:
                count += 1
            result[i] = count
        return pd.Series(result, index=s.index)

    out["preprocessed_consecutive_bars"] = _consecutive_same_sign(body_sign_shifted)

    # ── Time-to-Touch Features ─────────────────────────────────────────────────
    # Vol Persistence: ATR(14) / ATR(200) — tempo krátkodobé vs. strukturální volatility.
    # < 1 = klidný trh (ATR14 pod průměrem), > 1 = zrychlení pohybu.
    atr200 = compute_atr(high, low, close, ATR_200_PERIOD)
    out["preprocessed_vol_persistence"] = (
        (atr14 / atr200.replace(0, np.nan)).astype("float32")
    )

    # Price Speed: (Close[T-1] - Close[T-10]) / ATR(14) — rychlost pohybu v jednotkách ATR.
    # Kladné = vzestupný impulz za 10 svíček, záporné = sestupný.
    # Leakage guard: close.shift(1) a close.shift(10) → pouze uzavřené svíčky.
    out["preprocessed_price_speed"] = (
        ((close.shift(1) - close.shift(10)) / safe_atr).astype("float32")
    )

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
    # News window: 1 = bar leží v rizikovém časovém pásmu zpráv (CET).
    # V backtestu (pokud FILTER_NEWS=True) tyto bary přeskočí.
    # 14:30–14:45 CET = 870–885 min,   16:00–16:15 CET = 960–975 min.
    news_mask = (
        ((cet_min_day >= 870) & (cet_min_day < 885))  # 14:30–14:45 CET
        | ((cet_min_day >= 960) & (cet_min_day < 975))  # 16:00–16:15 CET
    )
    out["preprocessed_news_window"] = news_mask.astype("float32")
    # ── Smart Money Concepts (SMC) ─────────────────────────────────────────────
    # LEAKAGE GUARD: veškerá data z předchozích uzavřených svíček.
    #   close_t1     = close[T-1]  (uzavřená svíčka)
    #   daily_* .shift(1) → datum D čte hodnotu z D-1
    #   asia_*  .shift(1) → datum D čte asijský rozsah z D-1
    print("Počítám SMC features (PDH/PDL, FVG, Asia Range)...")

    close_t1     = close.shift(1)           # Close uzavřené svíčky T-1
    cet_date_arr = cet_idx.normalize()       # CET půlnoc pro každý M1 bar (tz-aware)

    # --- Previous Day High / Low (PDH / PDL) ----------------------------------
    # high.groupby(cet_date_arr) → jeden řádek na každý obchodní den (CET)
    # .shift(1) → datum D dostane H/L z D-1 → bez úniku dat
    daily_high = high.groupby(cet_date_arr).max()
    daily_low  = low.groupby(cet_date_arr).min()
    pdh_series = daily_high.shift(1)   # pro datum D → D-1 daily high
    pdl_series = daily_low.shift(1)    # pro datum D → D-1 daily low

    # Zpětné mapování na M1 pomocí CET date jako klíče
    date_mapper = pd.Series(cet_date_arr, index=df.index)
    pdh_m1 = date_mapper.map(pdh_series)
    pdl_m1 = date_mapper.map(pdl_series)

    # Vzdálenosti Close(T-1) od PDH/PDL v pipech (+ = nad úrovní, − = pod)
    out["preprocessed_pdh_dist"] = (
        ((close_t1 - pdh_m1) / PIP_SIZE).astype("float32")
    )
    out["preprocessed_pdl_dist"] = (
        ((close_t1 - pdl_m1) / PIP_SIZE).astype("float32")
    )

    # --- Fair Value Gap (FVG) -------------------------------------------------
    # Bullish FVG: High[T-2] < Low[T]  → kladný gap (Low[T] − High[T-2])
    # Bearish FVG: Low[T-2]  > High[T] → záporný gap (High[T] − Low[T-2])
    # Používáme uzavřená data svíčky T a T-2 → bez úniku dat.
    bull_fvg = (low  - high.shift(2)).clip(lower=0)   # > 0 pro bullish FVG
    bear_fvg = (high - low.shift(2)).clip(upper=0)    # < 0 pro bearish FVG
    out["preprocessed_fvg_size"] = (
        ((bull_fvg + bear_fvg) / PIP_SIZE).astype("float32")
    )

    # --- Asian Session Range Distances ----------------------------------------
    # Asijská seance: 00:00–09:00 CET (asia_mask definována výše v sekci Sessions)
    # Pro bar T → H/L asijské seance z PŘEDCHOZÍHO obchodního dne (D-1).
    high_asia_only  = high.where(asia_mask)          # NaN mimo asijskou seanci
    low_asia_only   = low.where(asia_mask)
    asia_daily_high = high_asia_only.groupby(cet_date_arr).max()
    asia_daily_low  = low_asia_only.groupby(cet_date_arr).min()
    asia_prev_high  = asia_daily_high.shift(1)       # D → D-1 Asia session high
    asia_prev_low   = asia_daily_low.shift(1)        # D → D-1 Asia session low

    asia_h_m1 = date_mapper.map(asia_prev_high)
    asia_l_m1 = date_mapper.map(asia_prev_low)

    out["preprocessed_asia_high_dist"] = (
        ((close_t1 - asia_h_m1) / PIP_SIZE).astype("float32")
    )
    out["preprocessed_asia_low_dist"] = (
        ((close_t1 - asia_l_m1) / PIP_SIZE).astype("float32")
    )

    # ── Market Regime Features ─────────────────────────────────────────────────────
    print("Počítám Market Regime features (ER, Vol Regime Z-score, Tick Log Ratio)...")

    # Efficiency Ratio (Kaufman ER): míra trendovosti.
    # ER = abs(directional_move) / sum(abs(step_i))
    # ER ≈ 1 → silný trend,  ER ≈ 0 → šum / konsolidace.
    # Leakage guard: close.shift(1)..close.shift(ER_WINDOW+1) → pouze uzavřené svíčky.
    directional = (close.shift(1) - close.shift(ER_WINDOW + 1)).abs()
    path_length = pd.concat(
        [(close.shift(k) - close.shift(k + 1)).abs() for k in range(1, ER_WINDOW + 1)],
        axis=1,
    ).sum(axis=1)
    out["preprocessed_er"] = (
        (directional / path_length.replace(0, np.nan)).clip(0, 1).astype("float32")
    )

    # Volatility Regime Z-score: Z-score ATR(200) vůči svému vlastnímu rolling průměru.
    # Říká modelu, zda je strukturní volatilita historicky vysoká nebo nízká.
    # Leakage guard: rolling(VOL_ZSCORE_WINDOW).shift(1) → okno [T-500…T-1].
    vol_regime_mean = (
        atr200.rolling(VOL_ZSCORE_WINDOW, min_periods=VOL_ZSCORE_WINDOW).mean().shift(1)
    )
    vol_regime_std = (
        atr200.rolling(VOL_ZSCORE_WINDOW, min_periods=VOL_ZSCORE_WINDOW).std().shift(1)
    )
    out["preprocessed_vol_regime_zscore"] = (
        (atr200 - vol_regime_mean) / vol_regime_std.replace(0, np.nan)
    ).astype("float32")

    # Tick Log Ratio: log(tick_volume / rolling_mean(tick_volume, 50)).
    # Logaritmická škála lépe zachytí extrémní objemové spiky než lineární volume_surge.
    # Leakage guard: rolling(50).mean().shift(1) → okno [T-50…T-1].
    vol_ma50 = volume.rolling(50, min_periods=50).mean().shift(1)
    out["preprocessed_tick_log_ratio"] = np.log(
        (volume / vol_ma50.replace(0, np.nan)).clip(1e-6, None)
    ).astype("float32")

    # ── Target + TBM outcome ─────────────────────────────────────────────────
    out["preprocessed_target"] = target
    out["tbm_outcome"]          = tbm_outcome   # 1=TP, 0=SL, 2=Timeout (není feature)

    # ── Základní OHLC + spread + ATR pro backtest (bez prefixu) ─────────────
    out["open"]   = open_.values
    out["high"]   = high.values
    out["low"]    = low.values
    out["close"]  = close.values
    out["spread"] = df["spread"].values   # raw spread v bodech (MT5 points)
    out["atr14"]  = atr14.values          # ATR(14) pro TBM simulaci v backtestu

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




