# EURUSD M1 ML Pipeline

Robustní machine-learning pipeline pro 1-minutová data EURUSD s absolutní eliminací data leakage.

---

## Struktura projektu

```
src/
  preprocess.py   – Feature engineering → data/eurusd_clean.parquet
  train.py        – LightGBM trénink + OOS evaluace → models/lgbm_clean_v1.pkl
  backtest.py     – Simulace obchodů na OOS datech → data/backtest_oos_results.parquet
  models.py       – Továrna modelů: get_model(name), get_config_params(name)
data/
  EURUSD_M1_full.parquet   – Surová M1 data (vstup)
  eurusd_clean.parquet     – Vyčištěná data s features (výstup preprocessingu)
models/
  lgbm_clean_v1.pkl        – Natrénovaný model
```

## Spuštění

```bash
python src/preprocess.py   # 1. Vytvořit features
python src/train.py        # 2. Trénovat model
python src/backtest.py     # 3. OOS backtest
```

---

## Přehled všech `preprocessed_` indikátorů

Všechny features mají prefix `preprocessed_`. Target (`preprocessed_target`) je **budoucnost** a nikdy nesmí vstoupit do X.

### Price Action

| Indikátor | Popis | Leakage guard |
|---|---|---|
| `preprocessed_body_atr` | Velikost těla svíčky (\|close−open\|) dělená ATR(14) | ATR počítá jen z uzavřených svíček |
| `preprocessed_upper_shadow_atr` | Horní stín (wick) dělený ATR(14) | viz výše |
| `preprocessed_lower_shadow_atr` | Dolní stín (wick) dělený ATR(14) | viz výše |
| `preprocessed_close_pos` | Pozice close v rámci High−Low kanálu svíčky (0 = dno, 1 = vršek) | jednobodový výpočet na uzavřené svíčce |

### Momentum

| Indikátor | Popis | Leakage guard |
|---|---|---|
| `preprocessed_rsi14` | RSI(14) – Wilderovo EWM (com=13) | min_periods=14 |
| `preprocessed_roc5` | Rate of Change close za 5 minut (`pct_change(5)`) | použito jen uzavřených svíček |
| `preprocessed_roc10` | Rate of Change za 10 minut | viz výše |
| `preprocessed_roc20` | Rate of Change za 20 minut | viz výše |

### Volatilita

| Indikátor | Popis | Leakage guard |
|---|---|---|
| `preprocessed_atr14_norm` | ATR(14) dělený close – normalizovaná volatilita | Wilderovo EWM, min_periods=14 |
| `preprocessed_vol_zscore` | Z-score ATR(14) vůči rolling(500) průměru/std – indikátor volatilitního režimu (mrtvý trh vs. news chaos) | rolling okno [t−500…t], bez budoucnosti |

### Trend (M1)

| Indikátor | Popis | Leakage guard |
|---|---|---|
| `preprocessed_log_close_ema50` | log(close / EMA(50)) | min_periods=50 |
| `preprocessed_log_close_ema200` | log(close / EMA(200)) | min_periods=200 |

### Multi-Timeframe Trend (H1)

| Indikátor | Popis | Leakage guard |
|---|---|---|
| `preprocessed_h1_trend` | log(close_M1 / H1_EMA200) – kde je cena vůči velkému trendu | H1 EMA200 je `.shift(1)` na H1 úrovni → model vidí pouze plně uzavřené H1 svíčky |
| `preprocessed_slope_ema200_h1` | Sklon H1 EMA200 za poslední 3 H1 hodiny = `(EMA200[T−1] − EMA200[T−4]) / 3` | odvozeno z již posunuté `h1_ema200_lagged`, zpět na M1 přes ffill |

### Volume

| Indikátor | Popis | Leakage guard |
|---|---|---|
| `preprocessed_volume_surge` | Poměr `tick_volume / mean_volume(20)` – detekuje abnormální aktivitu | `mean_volume(20)` je `.shift(1)` → průměr z [t−20…t−1] |

### Lagged Features (krátká paměť)

Lagy pro `body_atr`, `rsi14` a `volume_surge` dávají modelu přímý pohled na gradient — roste RSI? Graduje volume? Zrychluje se tělo svíčky?

| Indikátor | Popis |
|---|---|
| `preprocessed_body_atr_lag1` | `body_atr` z uzavřené svíčky T−1 |
| `preprocessed_body_atr_lag2` | `body_atr` z T−2 |
| `preprocessed_body_atr_lag3` | `body_atr` z T−3 |
| `preprocessed_body_atr_lag5` | `body_atr` z T−5 |
| `preprocessed_rsi14_lag1` | RSI(14) z T−1 |
| `preprocessed_rsi14_lag2` | RSI(14) z T−2 |
| `preprocessed_rsi14_lag3` | RSI(14) z T−3 |
| `preprocessed_rsi14_lag5` | RSI(14) z T−5 |
| `preprocessed_volume_surge_lag1` | volume_surge z T−1 |
| `preprocessed_volume_surge_lag2` | volume_surge z T−2 |
| `preprocessed_volume_surge_lag3` | volume_surge z T−3 |
| `preprocessed_volume_surge_lag5` | volume_surge z T−5 |

Všechny lagy používají `.shift(n)` → hodnota z uzavřené svíčky T−n, nikdy T.

### Rolling Trend Windows (dlouhá paměť)

| Indikátor | Popis | Leakage guard |
|---|---|---|
| `preprocessed_range_24h` | `max(high) − min(low)` za posledních 1 440 minut (24 h) | `.shift(1)` → okno [T−1440…T−1] |
| `preprocessed_close_vs_7d_highlow` | Pozice close v 7denním High−Low kanálu (0 = dno týdne, 1 = vrchol týdne) | `.shift(1)` na high/low rolling(10080) |

### Advanced Price Action

| Indikátor | Popis | Leakage guard |
|---|---|---|
| `preprocessed_is_high_20` | 1 pokud High[T−1] == max(High) za posledních 20 svíček – detekce swing high | `high.shift(1)` vs `rolling(20).max().shift(1)` |
| `preprocessed_is_low_20` | 1 pokud Low[T−1] == min(Low) za posledních 20 svíček – detekce swing low | analogicky |
| `preprocessed_atr_ratio` | ATR(14) / ATR(100) – poměr krátkodobé a dlouhodobé volatility; < 1 = komprese, > 1 = expanze | oba ATR počítány jen z uzavřených svíček |
| `preprocessed_dist_from_24h_high` | (Close[T−1] − 24h_High) v pipech – vzdálenost ceny od denního vrcholu | `close.shift(1)` a `high.rolling(1440).max().shift(1)` |
| `preprocessed_consecutive_bars` | Počet po sobě jdoucích svíček se stejným znaménkem těla (Up/Down) – momentum climax | počítáno z `sign(close−open).shift(1)` |

### Time & Sessions

| Indikátor | Popis |
|---|---|
| `preprocessed_session_london` | 1 pokud je svíčka v Londýnské seanci (08:00–17:00 CET/CEST) |
| `preprocessed_session_ny` | 1 pokud je svíčka v NY seanci (13:00–22:00 CET/CEST) |
| `preprocessed_session_asia` | 1 pokud je svíčka v Asijské seanci (00:00–09:00 CET/CEST) |
| `preprocessed_dist_london_open` | Počet minut od 08:00 CET (záporné = před otevřením Londýna) |
| `preprocessed_hour_sin` | Cyklické sin kódování minuty v rámci dne (perioda 1440 min) |
| `preprocessed_hour_cos` | Cyklické cos kódování minuty v rámci dne |
| `preprocessed_dow_sin` | Cyklické sin kódování dne v týdnu (0=Po…6=Ne) |
| `preprocessed_dow_cos` | Cyklické cos kódování dne v týdnu |

### Target (label – nikdy do features!)

| Indikátor | Popis |
|---|---|
| `preprocessed_target` | 1 pokud `close[T+1] > close[T]`, jinak 0. Poslední řádek je NaN. |

---

## Anti-Leakage pravidla

1. **Budoucnost**: `close.shift(-1)` je výhradně pro target — nikdy jako feature.
2. **Rolling okna**: vždy `.shift(1)` → okno končí v T−1.
3. **EWM (ATR, RSI, EMA)**: `min_periods` garantuje, že warmup fáze nevytváří NaN z částečných oken.
4. **H1 MTF**: H1 EMA je `.shift(1)` na H1 úrovni, pak `ffill` na M1 — model vidí jen plně uzavřené H1 svíčky.
5. **Prefixový filtr**: `train.py` přijme do modelu **výhradně** sloupce začínající `preprocessed_`. Runtime assert toto vynucuje.
6. **`dropna()` na konci**: odstraní warmup řádky (7d okno = 10 080 svíček) a poslední řádek bez targetu.


---

## Konfigurace modelu (`src/models.py`)

`get_model(name)` vrátí nakonfigurovaný `LGBMClassifier`. `get_config_params(name)` vrátí slovník parametrů pro logování.

| Konfigurace | Popis | Klíčové parametry |
|---|---|---|
| `default` | Vyvážený model pro maximální coverage | `n_estimators=1000, lr=0.01, max_depth=6, num_leaves=31` |
| `conservative` | Konzervativní model s L1/L2 regularizací | `n_estimators=1500, lr=0.005, max_depth=4, num_leaves=15, reg_alpha=0.1, reg_lambda=0.1` |

Přidání nové konfigurace: doplň záznam do `_CONFIGS` v `src/models.py`.

---

## Poslední OOS výsledky (model: `default`, test: 2025-01-01 →)

| Threshold | Precision Long | Recall Long | Coverage | N Long |
|---|---|---|---|---|
| 0.50 | 50.8% | 6.0% | 5.62% | 25 311 |
| 0.52 | 53.3% | 0.8% | 0.74% | 3 338 |
| 0.55 | 56.0% | 0.1% | 0.11% | 518 |
| 0.60 | 59.8% | 0.02% | 0.02% | 87 |

AUC (Long): **0.524** | Top feature: `preprocessed_atr14_norm`
