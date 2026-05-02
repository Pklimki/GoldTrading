"""
src/stress_test.py – Walk-Forward Stress Test (3 cykly).

Cykly:
  A: Train 2016–2022, Test OOS 2023
  B: Train 2016–2023, Test OOS 2024
  C: Train 2016–2024, Test OOS 2025+

Konfigurace:
  TBM:    PT = 3.0 × ATR(14),  SL = 2.0 × ATR(14),  Horizon = 24 M5 barů (120 min)
  Spread: 1.5 pip fixní
  Filtr:  preprocessed_is_tradable == 1

Spuštění (z kořene projektu):
    python src/stress_test.py
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.metrics import precision_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
import sys
import os

# Přidej src/ na cestu, aby fungovalo `from models import ...`
sys.path.insert(0, os.path.dirname(__file__))
from models import get_model, get_config_params

# ── Konstanty ─────────────────────────────────────────────────────────────────
DATA_PATH         = "data/eurusd_clean.parquet"
TARGET_COL        = "preprocessed_target"
MODEL_TYPE        = "default"
# Dynamický práh: vstupujeme jen do anomálií v distribuci predikce.
# thr[t] = rolling_mean(prob1, DYNAMIC_WINDOW) + DYNAMIC_SIGMA × rolling_std
DYNAMIC_WINDOW    = 500
DYNAMIC_SIGMA     = 2.0
TBM_HORIZON       = 24    # M5 barů (24 × 5 min = 120 min)
TBM_PT_MULT       = 3.0
TBM_SL_MULT       = 2.0
FIXED_SPREAD_PIPS = 1.5
PIP_VALUE         = 0.0001
FILTER_NEWS       = False  # True = přeskočí 14:30–14:45 a 16:00–16:15 CET

# Walk-forward cykly
CYCLES = [
    {
        "name":       "A",
        "desc":       "Train 2016–2022  |  Test 2023",
        "train_end":  pd.Timestamp("2022-12-31 23:59:59", tz="UTC"),
        "test_start": pd.Timestamp("2023-01-01 00:00:00", tz="UTC"),
        "test_end":   pd.Timestamp("2023-12-31 23:59:59", tz="UTC"),
    },
    {
        "name":       "B",
        "desc":       "Train 2016–2023  |  Test 2024",
        "train_end":  pd.Timestamp("2023-12-31 23:59:59", tz="UTC"),
        "test_start": pd.Timestamp("2024-01-01 00:00:00", tz="UTC"),
        "test_end":   pd.Timestamp("2024-12-31 23:59:59", tz="UTC"),
    },
    {
        "name":       "C",
        "desc":       "Train 2016–2024  |  Test 2025+",
        "train_end":  pd.Timestamp("2024-12-31 23:59:59", tz="UTC"),
        "test_start": pd.Timestamp("2025-01-01 00:00:00", tz="UTC"),
        "test_end":   None,
    },
]


# ── TBM simulace ──────────────────────────────────────────────────────────────

def simulate_tbm(oos: pd.DataFrame, sig_mask: np.ndarray, horizon: int = 0) -> pd.DataFrame:
    """
    Vektorizovaná TBM simulace Long obchodů.
    Vrátí DataFrame s jedním řádkem na obchod.
    horizon=0 použije modulální konstantu TBM_HORIZON.
    """
    h = horizon if horizon > 0 else TBM_HORIZON
    n = len(oos)
    m = n - h   # bary s plným předním oknem

    if m <= 0 or not sig_mask[:m].any():
        return pd.DataFrame()

    close_arr = oos["close"].values
    open_arr  = oos["open"].values
    high_arr  = oos["high"].values
    low_arr   = oos["low"].values
    atr14_arr = oos["atr14"].values

    win_h = np.lib.stride_tricks.sliding_window_view(
        high_arr, h + 1
    )[:, 1:]   # (m, h)
    win_l = np.lib.stride_tricks.sliding_window_view(
        low_arr, h + 1
    )[:, 1:]

    tp_level     = close_arr[:m] + atr14_arr[:m] * TBM_PT_MULT
    sl_level     = close_arr[:m] - atr14_arr[:m] * TBM_SL_MULT
    timeout_exit = close_arr[h:]   # close[T + h]

    sig_idx = np.where(sig_mask[:m])[0]

    wh     = win_h[sig_idx]
    wl     = win_l[sig_idx]
    tp_lev = tp_level[sig_idx]
    sl_lev = sl_level[sig_idx]
    to_cls = timeout_exit[sig_idx]

    entry_price = open_arr[sig_idx + 1]
    spread_pips = np.full(len(sig_idx), FIXED_SPREAD_PIPS)

    tp_hits  = wh >= tp_lev[:, None]
    sl_hits  = wl <= sl_lev[:, None]

    tp_first = np.where(tp_hits.any(axis=1), tp_hits.argmax(axis=1), h)
    sl_first = np.where(sl_hits.any(axis=1), sl_hits.argmax(axis=1), h)

    outcome = np.select(
        [tp_first < sl_first, sl_first < h],
        [1, 0], default=2,
    )
    exit_price = np.select(
        [outcome == 1, outcome == 0],
        [tp_lev, sl_lev], default=to_cls,
    )

    gross_pips = (exit_price - entry_price) / PIP_VALUE
    net_pips   = gross_pips - spread_pips

    return pd.DataFrame({
        "timestamp":  oos.index[sig_idx],
        "outcome":    outcome,    # 1=TP, 0=SL, 2=Timeout
        "net_pips":   net_pips,
        "atr14":      atr14_arr[sig_idx],
    }).set_index("timestamp")


# ── Běh jednoho cyklu ─────────────────────────────────────────────────────────

def run_cycle(df: pd.DataFrame, cycle: dict, feat_cols: list, tbm_horizon: int = 0) -> dict:
    train_end  = cycle["train_end"]
    test_start = cycle["test_start"]
    test_end   = cycle["test_end"]

    train_mask = df.index <= train_end
    test_mask  = (df.index >= test_start) & (
        (df.index <= test_end) if test_end else True
    )

    X_train = df.loc[train_mask, feat_cols]
    y_train = df.loc[train_mask, TARGET_COL].astype(int)
    X_test  = df.loc[test_mask,  feat_cols]
    y_test  = df.loc[test_mask,  TARGET_COL].astype(int)

    print(f"\n  Train: {train_mask.sum():,} barů  |  Test: {test_mask.sum():,} barů")
    print(f"  Target distribuce (train): {y_train.value_counts().to_dict()}")
    print(f"  Target distribuce (test):  {y_test.value_counts().to_dict()}")

    # Trénink
    model = get_model(MODEL_TYPE)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=200),
        ],
    )

    # Platt Scaling – post-hoc kalibrace (posledních 15 % trn. dat)
    cal_size     = max(5_000, int(len(X_train) * 0.15))
    X_cal, y_cal = X_train.iloc[-cal_size:], y_train.iloc[-cal_size:]
    raw_cal      = model.predict_proba(X_cal)[:, 1].reshape(-1, 1)
    platt        = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    platt.fit(raw_cal, y_cal)

    # Predikce (kalibrované pravděpodobnosti)
    raw_test = model.predict_proba(X_test)[:, 1].reshape(-1, 1)
    prob1    = platt.predict_proba(raw_test)[:, 1]
    y_pred   = (prob1 >= 0.5).astype(int)
    oos_df   = df.loc[test_mask].copy()

    auc   = roc_auc_score(y_test, prob1)
    prec  = precision_score(y_test, y_pred, pos_label=1, zero_division=0)

    # Diagnostika distribuce pravděpodobností
    pcts = np.percentile(prob1, [1, 5, 25, 50, 75, 95, 99])
    print(f"\n  prob1 distribuce: min={prob1.min():.4f}  mean={prob1.mean():.4f}  "
          f"max={prob1.max():.4f}  AUC={auc:.4f}")
    print(f"  Percentily [1,5,25,50,75,95,99]: "
          + "  ".join(f"p{p:.0f}={v:.4f}" for p, v in zip([1,5,25,50,75,95,99], pcts)))

    # Dynamický práh: obchodujeme jen anomálie v distribuci predikce.
    # thr[t] = rolling_mean(prob1, DYNAMIC_WINDOW) + DYNAMIC_SIGMA × rolling_std(prob1)
    prob1_s   = pd.Series(prob1)
    roll_mean = prob1_s.rolling(DYNAMIC_WINDOW, min_periods=50).mean().values
    roll_std  = prob1_s.rolling(DYNAMIC_WINDOW, min_periods=50).std().fillna(0).values
    dyn_thr   = roll_mean + DYNAMIC_SIGMA * roll_std

    # Práhová tabulka – statické prahy pro srovnání + dynamický práh
    print(f"\n  Threshold table (cyklus {cycle['name']}):")
    print(f"  {'Thr':>5}  {'Prec':>7}  {'Coverage':>10}  {'N Long':>8}")
    for thr in [0.40, 0.42, 0.45, 0.47, 0.50, 0.52, 0.55, 0.60]:
        yt = (prob1 >= thr).astype(int)
        pt = precision_score(y_test, yt, pos_label=1, zero_division=0)
        nl = int((yt == 1).sum())
        cov = nl / len(y_test) * 100
        print(f"  {thr:>5.2f}  {pt:>7.4f}  {cov:>9.2f}%  {nl:>8,}")
    dyn_n   = int((prob1 >= dyn_thr).sum())
    dyn_cov = dyn_n / len(y_test) * 100
    valid_thr = dyn_thr[~np.isnan(dyn_thr)]
    thr_mean = valid_thr.mean() if len(valid_thr) > 0 else float('nan')
    print(f"  {'dyn':>5}  {'—':>7}  {dyn_cov:>9.2f}%  {dyn_n:>8,}  (thr̄={thr_mean:.4f}, σ={DYNAMIC_SIGMA})")

    # Backtest simulace
    is_tradable = (oos_df["preprocessed_is_tradable"].values == 1) \
        if "preprocessed_is_tradable" in oos_df.columns \
        else np.ones(len(oos_df), dtype=bool)
    not_news = ~((oos_df["preprocessed_news_window"].values == 1) & FILTER_NEWS) \
        if "preprocessed_news_window" in oos_df.columns \
        else np.ones(len(oos_df), dtype=bool)

    sig_mask = (prob1 >= dyn_thr) & (prob1 > 0.50) & is_tradable & not_news
    trades   = simulate_tbm(oos_df, sig_mask, horizon=tbm_horizon)

    if len(trades) == 0:
        print("  Žádné simulované obchody.")
        return {"cycle": cycle["name"], "n_trades": 0}

    n      = len(trades)
    n_tp   = (trades["outcome"] == 1).sum()
    n_sl   = (trades["outcome"] == 0).sum()
    wins   = (trades["net_pips"] > 0).sum()
    avg    = trades["net_pips"].mean()
    total  = trades["net_pips"].sum()
    wr     = wins / n * 100
    tp_pct = n_tp / n * 100
    sl_pct = n_sl / n * 100

    print(f"\n  Backtest (spread={FIXED_SPREAD_PIPS} pip, thr=dyn σ={DYNAMIC_SIGMA}, horizon={tbm_horizon or TBM_HORIZON}):")
    print(f"    N obchodů   : {n:,}")
    print(f"    TP/SL       : TP={tp_pct:.1f}%  SL={sl_pct:.1f}%  Timeout={(1-tp_pct/100-sl_pct/100)*100:.1f}%")
    print(f"    Win Rate    : {wr:.2f}%")
    print(f"    Celkem P&L  : {total:+.2f} pips")
    print(f"    Avg Net Pips: {avg:+.4f} pips/trade  ← klíčová metrika")
    print(f"    Průměr ATR  : {trades['atr14'].mean()/PIP_VALUE:.2f} pips při vstupu")

    # Feature Importance (z primárního LGB modelu, ne wrapperu)
    fi = pd.Series(
        model.feature_importances_,
        index=feat_cols,
    ).sort_values(ascending=False)
    print(f"\n  Top-15 Feature Importance (cyklus {cycle['name']}, gain):")
    max_fi = fi.iloc[0] if len(fi) > 0 else 1
    for fname, fval in fi.head(15).items():
        bar = "█" * max(1, int(fval / max_fi * 25))
        print(f"    {fname:<44} {fval:>6}  {bar}")

    return {
        "cycle":    cycle["name"],
        "desc":     cycle["desc"],
        "auc":      auc,
        "prec50":   prec,
        "n_trades": n,
        "tp_pct":   tp_pct,
        "win_rate": wr,
        "total_pips": total,
        "avg_pips": avg,
        "feature_importances": fi,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("  STRESS TEST – Walk-Forward Validation (3 cykly)")
    print("=" * 70)
    print(f"  TBM: PT=ATR×{TBM_PT_MULT}  SL=ATR×{TBM_SL_MULT}  "
          f"Horizon={TBM_HORIZON} M5 barů  Spread={FIXED_SPREAD_PIPS} pip")
    print(f"  Filtr: is_tradable  |  FILTER_NEWS={'ON' if FILTER_NEWS else 'OFF'}")
    print(f"  Práh: DYNAMICKÝ (rolling {DYNAMIC_WINDOW} barů + {DYNAMIC_SIGMA}σ)")

    print("\nNačítám data...")
    df = pd.read_parquet(DATA_PATH)
    print(f"Načteno {len(df):,} řádků. Rozsah: {df.index.min()} → {df.index.max()}")

    feat_cols = [
        c for c in df.columns
        if c.startswith("preprocessed_") and c != TARGET_COL
    ]
    print(f"Features: {len(feat_cols)} sloupců")

    results = []
    for cycle in CYCLES:
        print(f"\n{'═' * 70}")
        print(f"  CYKLUS {cycle['name']}: {cycle['desc']}")
        print(f"{'═' * 70}")
        res = run_cycle(df, cycle, feat_cols)
        results.append(res)

    # ── Souhrnná tabulka ──────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  SOUHRN VÝSLEDKŮ")
    print(f"{'=' * 70}")
    hdr = f"  {'Cyklus':<6}  {'Popis':<30}  {'AUC':>6}  {'Prec50':>7}  "
    hdr += f"{'N':>7}  {'Win%':>6}  {'AvgPips':>9}  {'TotalPips':>10}"
    print(hdr)
    print("  " + "-" * 66)
    for r in results:
        if r["n_trades"] == 0:
            print(f"  {r['cycle']:<6}  — žádné obchody")
            continue
        line = (
            f"  {r['cycle']:<6}  {r['desc']:<30}  "
            f"{r['auc']:>6.4f}  {r['prec50']:>7.4f}  "
            f"{r['n_trades']:>7,}  {r['win_rate']:>5.1f}%  "
            f"{r['avg_pips']:>+9.4f}  {r['total_pips']:>+10.2f}"
        )
        print(line)
    print(f"{'=' * 70}")

    # ── Feature Importance – Agregace přes cykly ─────────────────────────────
    fi_list = [
        r["feature_importances"] for r in results
        if r.get("feature_importances") is not None
    ]
    if fi_list:
        fi_mean = (
            pd.concat(fi_list, axis=1)
            .mean(axis=1)
            .sort_values(ascending=False)
        )
        print(f"\n{'=' * 70}")
        print("  FEATURE IMPORTANCE – Průměr přes všechny cykly (Top 20)")
        print(f"{'=' * 70}")
        max_fi = fi_mean.iloc[0] if len(fi_mean) > 0 else 1
        for fname, fval in fi_mean.head(20).items():
            # Zvýrazni SMC skupinu
            tag = " ★" if any(k in fname for k in ("pdh", "pdl", "asia", "fvg")) else ""
            bar = "█" * max(1, int(fval / max_fi * 30))
            print(f"  {fname:<44} {fval:>7.1f}  {bar}{tag}")
        print(f"{'=' * 70}")

    # ── M15 Bonus Test (Cyklus C: 2025+) ─────────────────────────────────────
    m15_path = "data/eurusd_clean_15m.parquet"
    if os.path.exists(m15_path):
        print(f"\n{'═' * 70}")
        print(f"  M15 BONUS TEST – Cyklus C: Train 2016–2024  |  Test 2025+")
        print(f"  TBM Horizon = 12 M15 barů (12 × 15 min = 180 min)")
        print(f"{'═' * 70}")
        df_m15       = pd.read_parquet(m15_path)
        feat_m15     = [c for c in df_m15.columns
                        if c.startswith("preprocessed_") and c != TARGET_COL]
        print(f"  M15 data: {len(df_m15):,} řádků, {len(feat_m15)} features")
        res_m15 = run_cycle(df_m15, CYCLES[2], feat_m15, tbm_horizon=12)
        if res_m15.get("n_trades", 0) > 0:
            print(f"\n  M15 Cyklus C: {res_m15['n_trades']:,} obchodů  |  "
                  f"WR={res_m15['win_rate']:.1f}%  |  "
                  f"Avg Net Pips={res_m15['avg_pips']:+.4f}  |  "
                  f"Total={res_m15['total_pips']:+.2f}")
    else:
        print(f"\n  M15 data nenalezena ({m15_path}).")
        print(f"  Vygeneruj: nastav RESAMPLE_FREQ='15min' v src/preprocess.py a spusť znovu.")


if __name__ == "__main__":
    main()
