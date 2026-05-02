"""
src/backtest.py – OOS backtest se simulací Triple Barrier Method.

Logika:
  - Signál generuje uzavřená M1 svíčka T (predikce modelu na close[T]).
  - Vstup:  open svíčky T+1.
  - TP = close[T] + ATR(14)[T] * TBM_PT_MULT
  - SL = close[T] - ATR(14)[T] * TBM_SL_MULT
  - Horizon: TBM_HORIZON minut – po vypršení exit na close[T+HORIZON].
  - Spread odečten při vstupu: spread[T+1] / POINTS_PER_PIP (reálný spread z dat).
  - Jen OOS část dat (2025+).

Spuštění (z kořene projektu):
    python src/backtest.py
"""

import pandas as pd
import numpy as np
import joblib

DATA_PATH  = "data/eurusd_clean.parquet"
MODEL_PATH = "models/lgbm_clean_v1.pkl"

OOS_START        = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
TARGET_COL       = "preprocessed_target"

SIGNAL_THRESHOLD = 0.50   # pravděpodobnostní práh pro Long signál
TBM_HORIZON      = 60     # max počet svíček do exitu
TBM_PT_MULT      = 2.0    # TP = close[T] + ATR * TBM_PT_MULT
TBM_SL_MULT      = 1.5    # SL = close[T] - ATR * TBM_SL_MULT
POINTS_PER_PIP   = 10     # 10 MT5 bodů = 1 pip EURUSD
PIP_VALUE        = 0.0001  # 1 pip = 0.0001


def _simulate_tbm(
    oos: pd.DataFrame,
    sig_mask: np.ndarray,
) -> pd.DataFrame:
    """
    Vektorizovaná TBM simulace Long obchodů.

    Pro každý signál na baru T:
      - Entry = open[T+1]
      - TP    = close[T] + atr14[T] * TBM_PT_MULT
      - SL    = close[T] - atr14[T] * TBM_SL_MULT
      - Skenuje high/low v oknech T+1 … T+TBM_HORIZON
      - Timeout exit = close[T + TBM_HORIZON]
      - Spread (reálný z dat) = spread[T+1] / POINTS_PER_PIP odečten při vstupu

    Returns
    -------
    pd.DataFrame  s jedním řádkem na obchod.
    """
    n = len(oos)
    m = n - TBM_HORIZON   # bary s plným oknem (potřebujeme T+HORIZON existovat)

    close_arr  = oos["close"].values
    open_arr   = oos["open"].values
    high_arr   = oos["high"].values
    low_arr    = oos["low"].values
    spread_arr = oos["spread"].values
    atr14_arr  = oos["atr14"].values

    # Forward-looking windows: win_h[i, k] = high[i + k + 1], k = 0 … HORIZON-1
    win_h = np.lib.stride_tricks.sliding_window_view(
        high_arr, TBM_HORIZON + 1
    )[:, 1:]   # shape (m, TBM_HORIZON)
    win_l = np.lib.stride_tricks.sliding_window_view(
        low_arr, TBM_HORIZON + 1
    )[:, 1:]

    # TP/SL cenové úrovně (reference = close[T], ne entry price)
    tp_level     = close_arr[:m] + atr14_arr[:m] * TBM_PT_MULT   # (m,)
    sl_level     = close_arr[:m] - atr14_arr[:m] * TBM_SL_MULT
    timeout_exit = close_arr[TBM_HORIZON:]                         # close[T+HORIZON]

    # Maska signálů omezená na bary s plným oknem
    sig_m   = sig_mask[:m]
    if not sig_m.any():
        return pd.DataFrame()

    sig_idx = np.where(sig_m)[0]   # indexy signálů v OOS poli

    # Výběr dat jen pro signálové bary
    wh      = win_h[sig_idx]             # (n_sig, HORIZON)
    wl      = win_l[sig_idx]
    tp_lev  = tp_level[sig_idx]          # (n_sig,)
    sl_lev  = sl_level[sig_idx]
    to_cls  = timeout_exit[sig_idx]      # close[T+HORIZON] pro každý signál

    # Entry a spread
    entry_price = open_arr[sig_idx + 1]                    # open[T+1]
    spread_pips = spread_arr[sig_idx + 1] / POINTS_PER_PIP  # reálný spread z dat

    # Detekce prvního zasažení bariéry
    tp_hits  = wh >= tp_lev[:, None]   # (n_sig, HORIZON) bool
    sl_hits  = wl <= sl_lev[:, None]

    tp_first = np.where(tp_hits.any(axis=1), tp_hits.argmax(axis=1), TBM_HORIZON)
    sl_first = np.where(sl_hits.any(axis=1), sl_hits.argmax(axis=1), TBM_HORIZON)

    # Outcome: 1=TP, 0=SL, 2=Timeout
    outcome = np.select(
        [tp_first < sl_first,
         sl_first < TBM_HORIZON],
        [1, 0],
        default=2,
    )

    # Exit cena (Long trade)
    exit_price = np.select(
        [outcome == 1,   # TP zasažen
         outcome == 0],  # SL zasažen
        [tp_lev, sl_lev],
        default=to_cls,  # Timeout
    )

    # P&L v pipech: (exit - entry) / pip_value - spread
    gross_pips = (exit_price - entry_price) / PIP_VALUE
    net_pips   = gross_pips - spread_pips

    return pd.DataFrame({
        "timestamp":   oos.index[sig_idx],
        "entry_price": entry_price,
        "exit_price":  exit_price,
        "tp_level":    tp_lev,
        "sl_level":    sl_lev,
        "atr14":       atr14_arr[sig_idx],
        "spread_pips": spread_pips,
        "outcome":     outcome,        # 1=TP, 0=SL, 2=Timeout
        "gross_pips":  gross_pips,
        "net_pips":    net_pips,
    }).set_index("timestamp")


def _print_stats(trades: pd.DataFrame, label: str) -> None:
    if len(trades) == 0:
        print(f"\n{label}: žádné obchody")
        return
    n      = len(trades)
    n_tp   = (trades["outcome"] == 1).sum()
    n_sl   = (trades["outcome"] == 0).sum()
    n_to   = (trades["outcome"] == 2).sum()
    wins   = (trades["net_pips"] > 0).sum()
    total  = trades["net_pips"].sum()
    avg    = trades["net_pips"].mean()
    med    = trades["net_pips"].median()
    cum    = trades["net_pips"].cumsum()
    dd     = (cum - cum.cummax()).min()
    wr     = wins / n * 100

    print(f"\n{'─' * 62}")
    print(f"  {label}")
    print(f"{'─' * 62}")
    print(f"  Počet obchodů              : {n:>10,}")
    print(f"  TBM výsledky               :  "
          f"TP={n_tp:,} ({n_tp/n*100:.1f}%)  "
          f"SL={n_sl:,} ({n_sl/n*100:.1f}%)  "
          f"Timeout={n_to:,} ({n_to/n*100:.1f}%)")
    print(f"  Win rate                   : {wr:>9.2f} %")
    print(f"  Průměrný spread / obchod   : {trades['spread_pips'].mean():>9.2f} pips")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  Celkové net P&L            : {total:>9.2f} pips")
    print(f"  Average Net Pips per Trade : {avg:>9.4f} pips  ← klíčová metrika")
    print(f"  Medián P&L / obchod        : {med:>9.4f} pips")
    print(f"  Max Drawdown               : {dd:>9.2f} pips")


def _sharpe(pnl: pd.Series) -> float:
    """Annualizovaný Sharpe (denní součty P&L, bez risk-free rate)."""
    daily = pnl.groupby(pnl.index.date).sum()
    std   = daily.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return daily.mean() / std * np.sqrt(252)


def main() -> None:
    print("Načítám data a model...")
    df    = pd.read_parquet(DATA_PATH)
    model = joblib.load(MODEL_PATH)

    # ── OOS filtrace ──────────────────────────────────────────────────────────
    oos = df.loc[df.index >= OOS_START].copy()
    print(f"OOS řádků: {len(oos):,}  ({oos.index.min()} → {oos.index.max()})")

    if "atr14" not in oos.columns:
        raise KeyError(
            "Sloupec 'atr14' nenalezen v parquet souboru.\n"
            "Spusť znovu: python src/preprocess.py"
        )

    # ── Generování signálů ─────────────────────────────────────────────────────
    feat_cols = [
        c for c in oos.columns
        if c.startswith("preprocessed_") and c != TARGET_COL
    ]
    X_oos    = oos[feat_cols]
    prob1    = model.predict_proba(X_oos)[:, 1]
    sig_mask = prob1 >= SIGNAL_THRESHOLD

    print(f"\nSignály (threshold={SIGNAL_THRESHOLD}): {sig_mask.sum():,} Long signálů "
          f"({sig_mask.mean()*100:.2f}% OOS barů)")

    # ── TBM simulace Long obchodů ─────────────────────────────────────────────
    print(f"Simuluji TBM obchody (horizon={TBM_HORIZON} min, "
          f"PT=ATR×{TBM_PT_MULT}, SL=ATR×{TBM_SL_MULT})...")
    trades = _simulate_tbm(oos, sig_mask)

    if len(trades) == 0:
        print("Žádné obchody s dostatečným budoucím oknem.")
        return

    print(f"Simulováno {len(trades):,} obchodů.")

    # ── Výsledky ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  BACKTEST VÝSLEDKY  (OOS: 2025-01-01 → konec)")
    print("=" * 62)
    _print_stats(trades, "Long signály (All)")

    # Breakdown podle thresholdů
    print("\n" + "=" * 62)
    print("  VÝSLEDKY PODLE PRAVDĚPODOBNOSTNÍHO PRAHU")
    print("=" * 62)
    for thr in [0.50, 0.52, 0.55, 0.60]:
        thr_mask    = prob1[:len(sig_mask)] >= thr
        sub_trades  = _simulate_tbm(oos, thr_mask)
        if len(sub_trades) == 0:
            continue
        avg = sub_trades["net_pips"].mean()
        n   = len(sub_trades)
        wr  = (sub_trades["net_pips"] > 0).mean() * 100
        n_tp = (sub_trades["outcome"] == 1).sum()
        print(f"  thr={thr:.2f}  N={n:>6,}  "
              f"TP={n_tp/n*100:5.1f}%  "
              f"WR={wr:5.1f}%  "
              f"Average Net Pips/Trade={avg:+.4f}")

    # ── Sharpe ────────────────────────────────────────────────────────────────
    sharpe = _sharpe(trades["net_pips"])
    print(f"\n  Sharpe (denní P&L, bez rf): {sharpe:.4f}")
    print(f"  Průměrný ATR při vstupu   : {trades['atr14'].mean()/0.0001:.2f} pips")

    # ── Uložení ───────────────────────────────────────────────────────────────
    out_path = "data/backtest_oos_results.parquet"
    trades.to_parquet(out_path)
    print(f"\nDetailní výsledky uloženy do '{out_path}'.")


if __name__ == "__main__":
    main()
