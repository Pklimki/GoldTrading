"""
src/backtest.py – OOS backtest verifikace ML signálů.

Logika:
  - Signál generuje uzavřená M1 svíčka t (predikce modelu na close[t]).
  - Vstup:  open svíčky t+1.
  - Výstup: close svíčky t+1.
  - Náklady: spread svíčky t+1 (raw MT5 bod → pips: spread / 10,
             protože EURUSD je kótován na 5 desetinných míst,
             10 points = 1 pip = 0.0001).
  - Jen OOS část dat (2025+).

Spuštění (z kořene projektu):
    python src/backtest.py
"""

import pandas as pd
import numpy as np
import joblib

DATA_PATH  = "data/eurusd_clean.parquet"
MODEL_PATH = "models/lgbm_clean_v1.pkl"

OOS_START  = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
TARGET_COL = "preprocessed_target"

# Konverze spreadu: 1 pip EURUSD = 0.0001 = 10 MT5 bodů
POINTS_PER_PIP = 10
PIP_VALUE      = 0.0001   # EURUSD: 1 pip = 0.0001


def main() -> None:
    print("Načítám data a model...")
    df    = pd.read_parquet(DATA_PATH)
    model = joblib.load(MODEL_PATH)

    # ── OOS filtrace ──────────────────────────────────────────────────────────
    oos = df.loc[df.index >= OOS_START].copy()
    print(f"OOS řádků: {len(oos):,}  ({oos.index.min()} → {oos.index.max()})")

    # ── Generování signálů (stejná pravidla jako train.py) ─────────────────────
    feat_cols = [
        c for c in oos.columns
        if c.startswith("preprocessed_") and c != TARGET_COL
    ]
    X_oos = oos[feat_cols]

    oos["pred"]  = model.predict(X_oos)
    oos["prob1"] = model.predict_proba(X_oos)[:, 1]

    # ── Příprava vstupních/výstupních cen (t+1) ────────────────────────────────
    # shift(-1): pro řádek t poskytne hodnotu řádku t+1
    oos["next_open"]   = oos["open"].shift(-1)
    oos["next_close"]  = oos["close"].shift(-1)
    oos["next_spread"] = oos["spread"].shift(-1)

    # Odstraň poslední řádek – nemá t+1
    oos = oos.dropna(subset=["next_open", "next_close", "next_spread"])

    # Spread v cenových jednotkách (EURUSD pips → price)
    # MT5 spread: 40 bodů = 40/10 = 4 pips = 4 * 0.0001 = 0.0004
    spread_price = (oos["next_spread"] / POINTS_PER_PIP) * PIP_VALUE
    spread_pips  = oos["next_spread"] / POINTS_PER_PIP

    # ── Výpočet P&L ───────────────────────────────────────────────────────────
    # Long (pred=1):  buy open[t+1], sell close[t+1]
    # Short (pred=0): sell open[t+1], buy close[t+1]
    gross_pnl = np.where(
        oos["pred"] == 1,
        oos["next_close"] - oos["next_open"],     # long
        oos["next_open"]  - oos["next_close"],    # short
    )
    # Spread se platí při vstupu (buy-side spread)
    net_pnl_price = gross_pnl - spread_price.values
    net_pnl_pips  = net_pnl_price / PIP_VALUE

    oos["net_pnl_pips"] = net_pnl_pips

    # ── Agregované statistiky ─────────────────────────────────────────────────
    long_mask  = oos["pred"] == 1
    short_mask = oos["pred"] == 0

    def stats(mask: pd.Series, label: str) -> None:
        sub   = oos.loc[mask, "net_pnl_pips"]
        n     = len(sub)
        if n == 0:
            print(f"\n{label}: žádné obchody")
            return
        wins  = (sub > 0).sum()
        total = sub.sum()
        avg   = sub.mean()
        med   = sub.median()
        dd    = sub.cumsum().sub(sub.cumsum().cummax()).min()
        wr    = wins / n * 100

        print(f"\n{'─'*50}")
        print(f"  {label}")
        print(f"{'─'*50}")
        print(f"  Počet obchodů  : {n:>10,}")
        print(f"  Win rate       : {wr:>10.2f} %")
        print(f"  Celkové P&L    : {total:>10.2f} pips")
        print(f"  Průměr / obchod: {avg:>10.4f} pips")
        print(f"  Medián / obchod: {med:>10.4f} pips")
        print(f"  Max Drawdown   : {dd:>10.2f} pips")

    print("\n" + "=" * 50)
    print("  BACKTEST VÝSLEDKY  (OOS: 2025-01-01 → konec)")
    print("=" * 50)
    stats(long_mask,  "Long  (pred=1)")
    stats(short_mask, "Short (pred=0)")
    stats(pd.Series(True, index=oos.index), "Celkem (All)")

    # ── Ekvitní křivka: kumulativní P&L ──────────────────────────────────────
    equity = oos["net_pnl_pips"].cumsum()
    sharpe_daily = _sharpe(oos["net_pnl_pips"])
    print(f"\n  Sharpe (denní P&L, bez rf): {sharpe_daily:.4f}")
    print(f"  Průměrný spread          : {spread_pips.mean():.2f} pips")

    # ── Uložení výsledků ──────────────────────────────────────────────────────
    out_path = "data/backtest_oos_results.parquet"
    oos[["pred", "prob1", "net_pnl_pips"]].to_parquet(out_path)
    print(f"\nDetailní výsledky uloženy do '{out_path}'.")


def _sharpe(pnl: pd.Series, periods_per_day: int = 1440) -> float:
    """Annualizovaný Sharpe na základě M1 P&L série (bez risk-free rate)."""
    daily = pnl.groupby(pnl.index.date).sum()
    if daily.std() == 0:
        return 0.0
    return daily.mean() / daily.std() * np.sqrt(252)


if __name__ == "__main__":
    main()
