# EURUSD M1 ML Pipeline

Robustní machine-learning pipeline pro 1-minutová data EURUSD s absolutní eliminací data leakage.

---

## Struktura projektu

```
src/
  preprocess.py   – Feature engineering → data/eurusd_clean.parquet (M5) nebo eurusd_clean_15m.parquet (M15)
  train.py        – LightGBM trénink + Platt Scaling kalibrace + OOS evaluace → models/lgbm_clean_v1.pkl
  backtest.py     – Simulace obchodů na OOS datech → data/backtest_oos_results.parquet
  stress_test.py  – Walk-forward stress test (3 cykly + M15 bonus)
  models.py       – Továrna modelů: get_model(name), get_config_params(name)
data/
  EURUSD_M1_full.parquet      – Surová M1 data (vstup)
  eurusd_clean.parquet        – M5 data s features (665 514 řádků, 62 sloupců)
  eurusd_clean_15m.parquet    – M15 data s features (218 368 řádků, 62 sloupců)
models/
  lgbm_clean_v1.pkl           – Natrénovaný model (dict: {"lgbm": model, "platt": platt})
```

## Spuštění

```bash
# Preprocessing – M5 (výchozí)
python src/preprocess.py

# Preprocessing – M15 (nastavit RESAMPLE_FREQ="15min" v preprocess.py)
python src/preprocess.py

python src/train.py        # Trénovat model + Platt Scaling
python src/backtest.py     # OOS backtest
python src/stress_test.py  # Walk-forward stress test (3 cykly + M15 bonus)
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
| `preprocessed_vol_regime_zscore` | Z-score ATR(200) vůči rolling(500) průměru/std – dlouhodobý volatilitní režim | `atr200.shift(1)` a rolling(500) z [t−501…t−1] |
| `preprocessed_er` | Efficiency Ratio (Kaufman ER) za 10 svíček: pohyb/šum – detekuje trending vs. choppy trh | `close.shift(1..11)` – bez aktuální svíčky |
| `preprocessed_tick_log_ratio` | log(tick_volume / rolling_mean(tick_volume, 50)) – relativní aktivita obchodníků vůči baseline | `rolling(50).mean().shift(1)` |

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
| `default` | Vyvážený model pro maximální coverage | `n_estimators=1000, lr=0.01, max_depth=6, num_leaves=31, min_child_samples=50` |
| `conservative` | Konzervativní model s L1/L2 regularizací | `n_estimators=1500, lr=0.005, max_depth=4, num_leaves=15, reg_alpha=0.1, reg_lambda=0.1` |

Přidání nové konfigurace: doplň záznam do `_CONFIGS` v `src/models.py`.

---

## Kalibrace pravděpodobností (Platt Scaling)

Po tréningu LightGBM je model kalibrován pomocí **Platt Scaling** (logistická regrese na raw LGB pravděpodobnostech).

- Kalibrační sada: posledních 15 % tréninku (min. 5 000 vzorků)
- Kalibruje distribuci od `[0.21, 0.58]` na `[0.06, 0.84]` – model generuje ostřejší anomálie
- Implementace: `sklearn.linear_model.LogisticRegression(C=1.0)` na raw probs z train sady
- Uloženo jako dict `{"lgbm": model, "platt": platt}` v `lgbm_clean_v1.pkl`

> **Poznámka**: `CalibratedClassifierCV(cv='prefit')` byl odstraněn ze sklearn 1.3+. Vždy použij manuální Platt Scaling.

---

## Dynamický práh signálu

Místo fixního prahu se signál generuje pouze při **statistické anomálii** v distribuci predikce:

```
signal[t] = prob1[t] > rolling_mean(prob1, 500) + 1.5 × rolling_std(prob1, 500)
```

- Adaptuje se na změny distribuce modelu v čase (market regime shifts)
- `DYNAMIC_WINDOW=500`, `DYNAMIC_SIGMA=1.5` v `src/stress_test.py`
- Generuje ~7–9 % coverage (vs. fixní 0.47 → 3.9 %)

---

## M15 podpora

`src/preprocess.py` podporuje přepnutí časového rámce přes konstantu `RESAMPLE_FREQ`:

```python
RESAMPLE_FREQ = "5min"   # M5 (výchozí) → eurusd_clean.parquet
RESAMPLE_FREQ = "15min"  # M15          → eurusd_clean_15m.parquet
```

- `TBM_HORIZON` se automaticky přizpůsobí: M5=24 barů (2 h), M15=8 barů (2 h)
- `stress_test.py` spouští M15 bonus cyklus na `eurusd_clean_15m.parquet` automaticky (pokud soubor existuje)

---

## Walk-Forward Stress Test (`src/stress_test.py`)

Tři opakované cykly trénink → test bez data leakage:

| Cyklus | Trénink | Test | AUC | N obchodů | Win% | Avg Pips |
|---|---|---|---|---|---|---|
| A | 2016–2022 | 2023 | 0.6072 | 5 264 | 41.0% | −1.37 |
| B | 2016–2023 | 2024 | 0.6072 | 3 238 | 38.3% | −1.58 |
| C | 2016–2024 | 2025+ | 0.6123 | 6 541 | 41.0% | −1.20 |
| M15 C | 2016–2024 | 2025+ (M15) | 0.6994 | 2 749 | 42.5% | −1.02 |

TBM parametry: PT=3×ATR, SL=2×ATR, Spread=1.5 pip, Horizon=24 barů (M5) / 8 barů (M15).  
Práh: dynamický (rolling 500 + 1.5σ). M15 dosahuje AUC 0.70, výrazně lepší separace.

**Top-5 features (průměr přes cykly):** `range_24h`, `close_vs_7d_highlow`, `atr14_norm`, `dist_london_open`, **`vol_regime_zscore`** (nová #5)

---

## Poslední OOS výsledky (stress test walk-forward, dynamický práh σ=1.5)

| Cyklus | Test | AUC | N obchodů | Win% | Avg Pips | Total Pips |
|---|---|---|---|---|---|---|
| A | 2023 | 0.6072 | 5 264 | 41.0% | −1.37 | −7 232 |
| B | 2024 | 0.6072 | 3 238 | 38.3% | −1.58 | −5 122 |
| C | 2025+ | 0.6123 | 6 541 | 41.0% | −1.20 | −7 852 |
| M15 C | 2025+ (M15) | 0.6994 | 2 749 | 42.5% | −1.02 | −2 807 |

Výsledky ukazují konzistentní AUC ~0.61 napříč cykly. Statické prahy 0.55–0.60 dosahují precision 37–66 % (cyklus B thr=0.60: precision=66 %), ale velmi nízkou coverage. Dynamický práh generuje více obchodů, ale stále nedosahuje breakeven (nutná precision >60 % pro R:R=3:2).
