"""
backtest.py

Vektorizovaný backtest modelu na OOS datech (2024+).

Logika:
  - Každá svíčka s P(Long) >= long_thr otevře Long pozici.
  - Každá svíčka s P(Long) <= short_thr otevře Short pozici.
  - Exit po pevném časovém horizontu odpovídajícím 1 hodině (label_h1).
    Počet kroků = round(60 / medián_frekvence_v_minutách).
  - P&L = rozdíl Close(vstup) vs Close(exit) v pipech.
  - Každý obchod je zatížen 1.0 pipem (spread 0.8 + slippage 0.2).

Použití:
    python backtest.py --data data/eurusd.parquet --model models/lgbm_model.pkl
    python backtest.py --data data/eurusd.parquet --model models/lgbm_model.pkl \\
        --long-threshold 0.53 --short-threshold 0.46 --pip-size 0.0001
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import load_and_prepare
from src.features import add_engineered_features, select_priority_features, prepare_Xy
from src.model import load_model


# ---------------------------------------------------------------------------
# Pomocné funkce
# ---------------------------------------------------------------------------

def detect_bar_frequency(index: pd.DatetimeIndex) -> int:
    """
    Odhadne medián frekvence svíček v minutách.
    Vrátí počet kroků odpovídající 1 hodině (exit horizon).
    """
    diffs = index.to_series().diff().dropna()
    median_minutes = diffs.median().total_seconds() / 60
    if median_minutes <= 0:
        median_minutes = 5  # fallback
    steps = max(1, round(60 / median_minutes))
    return steps, median_minutes


def run_backtest(
    df_raw: pd.DataFrame,
    model,
    feature_cols: list,
    long_threshold: float = 0.53,
    short_threshold: float = 0.46,
    pip_size: float = 0.0001,
    cost_pips: float = 1.0,
) -> pd.DataFrame:
    """
    Vektorizovaný backtest. Vrátí DataFrame se všemi signály a P&L.

    Parametry
    ---------
    df_raw        : testovací DataFrame (musí obsahovat sloupec 'close' a 'target')
    model         : natrénovaný klasifikátor s predict_proba()
    feature_cols  : seznam feature sloupců použitých při tréninku
    long_threshold: P(Long) >= X → Long signál
    short_threshold: P(Long) <= X → Short signál
    pip_size      : velikost 1 pipu v ceně (0.0001 pro EURUSD, 0.01 pro XAUUSD)
    cost_pips     : celkové náklady na obchod v pipech (spread + slippage)
    """
    # Připrav features
    X, y = prepare_Xy(df_raw, feature_cols)
    close = df_raw["close"].reindex(X.index)

    # Detekuj frekvenci a spočítej exit horizont
    steps, med_min = detect_bar_frequency(close.index)
    print(f"[backtest] Medián frekvence: {med_min:.1f} min  →  exit po {steps} svíčkách (≈ 1 hod)")

    # Predikce pravděpodobností
    proba = model.predict_proba(X)[:, 1]

    n = len(close)
    close_arr = close.to_numpy()
    idx = close.index

    # Vektorizovaně přiřaď exit price:
    # exit_idx = min(i + steps, n-1) pro každou pozici i
    exit_positions = np.minimum(np.arange(n) + steps, n - 1)
    exit_close = close_arr[exit_positions]
    entry_close = close_arr

    # P&L v pipech (před náklady)
    raw_pnl_long  = (exit_close - entry_close) / pip_size   # Long: zisk při růstu
    raw_pnl_short = (entry_close - exit_close) / pip_size   # Short: zisk při poklesu

    # Signály
    long_mask  = proba >= long_threshold
    short_mask = proba <= short_threshold

    # Sestavit výsledkový DataFrame
    trades = pd.DataFrame(index=idx)
    trades["close_entry"]  = entry_close
    trades["close_exit"]   = exit_close
    trades["proba"]        = proba
    trades["signal"]       = 0
    trades.loc[long_mask,  "signal"] = 1
    trades.loc[short_mask, "signal"] = -1

    # P&L s náklady (pouze na signálech; ostatní řádky = 0)
    trades["pnl_pips"] = 0.0
    trades.loc[long_mask,  "pnl_pips"] = raw_pnl_long[long_mask]  - cost_pips
    trades.loc[short_mask, "pnl_pips"] = raw_pnl_short[short_mask] - cost_pips

    return trades


def print_summary(trades: pd.DataFrame) -> None:
    """Vypíše souhrnné statistiky backtestu."""
    signals = trades[trades["signal"] != 0].copy()
    n_total    = len(signals)
    n_long     = (signals["signal"] == 1).sum()
    n_short    = (signals["signal"] == -1).sum()
    n_win      = (signals["pnl_pips"] > 0).sum()
    win_rate   = n_win / n_total if n_total > 0 else 0
    total_pnl  = signals["pnl_pips"].sum()
    avg_pnl    = signals["pnl_pips"].mean()
    max_dd     = _max_drawdown(signals["pnl_pips"].cumsum())
    profit_factor = (
        signals.loc[signals["pnl_pips"] > 0, "pnl_pips"].sum()
        / abs(signals.loc[signals["pnl_pips"] < 0, "pnl_pips"].sum())
        if (signals["pnl_pips"] < 0).any() else float("inf")
    )
    sharpe = _sharpe(signals["pnl_pips"])

    print(f"\n{'='*52}")
    print(f"  Backtest – Výsledky (OOS data)")
    print(f"{'='*52}")
    print(f"  Celkem obchodů : {n_total:,}  (Long: {n_long:,}, Short: {n_short:,})")
    print(f"  Win rate        : {win_rate:.1%}")
    print(f"  Celkový P&L     : {total_pnl:+.1f} pipů")
    print(f"  Průměr / obchod : {avg_pnl:+.2f} pipů")
    print(f"  Profit Factor   : {profit_factor:.2f}")
    print(f"  Sharpe (denní)  : {sharpe:.2f}")
    print(f"  Max Drawdown    : {max_dd:.1f} pipů")
    print(f"{'='*52}\n")


def _max_drawdown(cumulative: pd.Series) -> float:
    """Maximální pokles od vrcholu v pipech."""
    rolling_max = cumulative.cummax()
    drawdown = cumulative - rolling_max
    return float(drawdown.min())


def _sharpe(pnl_series: pd.Series, periods_per_day: float = 288) -> float:
    """
    Annualizovaný Sharpe ratio.
    periods_per_day = 288 pro 5min svíčky (288 = 24*60/5).
    Pro M1 = 1440, H1 = 24. Použijeme počet obchodů/den.
    """
    if pnl_series.std() == 0:
        return 0.0
    mean = pnl_series.mean()
    std  = pnl_series.std()
    # Odhadni počet obchodů za rok
    return float((mean / std) * np.sqrt(252 * max(1, len(pnl_series) / 252)))


def plot_equity_curve(trades: pd.DataFrame, output_path: str | None = None) -> None:
    """Vykreslí kumulativní equity curve + měsíční P&L."""
    signals = trades[trades["signal"] != 0].copy()
    signals["cum_pnl"] = signals["pnl_pips"].cumsum()

    fig, axes = plt.subplots(2, 1, figsize=(16, 9),
                              gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle("Backtest – Equity Curve (OOS 2024+)", fontsize=14)

    # --- Horní graf: kumulativní P&L ---
    ax1 = axes[0]
    signals["cum_pnl"].plot(ax=ax1, color="steelblue", linewidth=1.2, label="Equity (pipy)")

    # Drawdown fill
    rolling_max = signals["cum_pnl"].cummax()
    ax1.fill_between(signals.index, signals["cum_pnl"], rolling_max,
                     alpha=0.25, color="red", label="Drawdown")

    # Rozlišení Long/Short barevnými body (podvzorkování kvůli rychlosti)
    sample_step = max(1, len(signals) // 2000)
    longs  = signals[signals["signal"] == 1].iloc[::sample_step]
    shorts = signals[signals["signal"] == -1].iloc[::sample_step]
    ax1.scatter(longs.index,  longs["cum_pnl"],  color="green", s=4, alpha=0.4, label="Long")
    ax1.scatter(shorts.index, shorts["cum_pnl"], color="red",   s=4, alpha=0.4, label="Short")

    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax1.set_ylabel("Kumulativní P&L (pipy)")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.4)

    # --- Dolní graf: měsíční P&L ---
    ax2 = axes[1]
    monthly = signals["pnl_pips"].resample("ME").sum()
    colors  = ["#2ecc71" if v >= 0 else "#e74c3c" for v in monthly]
    monthly.plot.bar(ax=ax2, color=colors, width=0.8)
    ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_xlabel("")
    ax2.set_ylabel("Měsíční P&L (pipy)")
    ax2.set_xticklabels([t.strftime("%b %y") for t in monthly.index], rotation=45, fontsize=7)
    ax2.grid(True, alpha=0.4, axis="y")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[backtest] Graf uložen → {output_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest ML modelu na OOS datech")
    p.add_argument("--data",             type=str,   default="data/eurusd.parquet",
                   help="Cesta k datovému souboru (.parquet nebo .csv)")
    p.add_argument("--model",            type=str,   default="models/lgbm_model.pkl",
                   help="Cesta k uloženému modelu (.pkl)")
    p.add_argument("--test-start",       type=str,   default="2024-01-01",
                   help="Začátek OOS testu (default: 2024-01-01)")
    p.add_argument("--train-end",        type=str,   default="2023-12-31",
                   help="Konec trénovacích dat (default: 2023-12-31)")
    p.add_argument("--long-threshold",   type=float, default=0.53,
                   help="P(Long) >= X → Long (default: 0.53)")
    p.add_argument("--short-threshold",  type=float, default=0.46,
                   help="P(Long) <= X → Short (default: 0.46)")
    p.add_argument("--pip-size",         type=float, default=0.0001,
                   help="Velikost 1 pipu v ceně (0.0001 pro EURUSD, 0.01 pro XAUUSD)")
    p.add_argument("--cost-pips",        type=float, default=1.0,
                   help="Náklady na obchod v pipech (spread + slippage, default: 1.0)")
    p.add_argument("--save-chart",       type=str,   default=None,
                   help="Cesta pro uložení grafu (např. reports/equity.png). Bez tohoto argumentu se graf zobrazí.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Načti a připrav data
    print(f"\n[backtest] Načítám data: {args.data}")
    print(f"[backtest] Train: do {args.train_end}  |  OOS test: od {args.test_start}")
    _, test_df, feature_cols = load_and_prepare(
        filepath=args.data,
        train_end=args.train_end,
        test_start=args.test_start,
    )

    # 2. Feature engineering (stejný pipeline jako při tréninku)
    test_df = add_engineered_features(test_df)
    # Omez feature_cols na to, co select_priority_features vrátí pro test_df
    feature_cols = select_priority_features(test_df)

    # 3. Načti model
    model = load_model(args.model)

    # Zkontroluj, že model zná stejné features
    if hasattr(model, "feature_name_"):
        model_features = list(model.feature_name_)
        missing = [f for f in model_features if f not in test_df.columns]
        extra   = [f for f in feature_cols if f not in model_features]
        if missing:
            print(f"[backtest] VAROVÁNÍ: {len(missing)} features chybí v datech: {missing[:5]}")
        # Použij feature_cols z modelu, ne z dat (zaručí konzistenci)
        feature_cols = [f for f in model_features if f in test_df.columns]

    # 4. Spusť backtest
    print(f"\n[backtest] Long threshold: >= {args.long_threshold}  |  Short threshold: <= {args.short_threshold}")
    print(f"[backtest] Pip size: {args.pip_size}  |  Náklady: {args.cost_pips} pipy/obchod")

    trades = run_backtest(
        df_raw=test_df,
        model=model,
        feature_cols=feature_cols,
        long_threshold=args.long_threshold,
        short_threshold=args.short_threshold,
        pip_size=args.pip_size,
        cost_pips=args.cost_pips,
    )

    # 5. Výsledky
    print_summary(trades)

    # 6. Graf
    save_chart = args.save_chart
    if save_chart:
        Path(save_chart).parent.mkdir(parents=True, exist_ok=True)
    plot_equity_curve(trades, output_path=save_chart)


if __name__ == "__main__":
    main()
