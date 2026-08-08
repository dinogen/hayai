# 03 — Feature engineering

Questo documento descrive, a livello di formula, **tutte le feature** calcolate da
`hayai_bo.py` e la **definizione del target** usato per il training.

Input di partenza: un dataframe con colonne `date`, `symbol`, `open`, `close`, `high`,
`low`, `volume` (da `hist/{symbol}.parquet`), processato **per simbolo** da
`add_time_features` e `add_financial_features`, poi aggregato e processato in modo
cross-sezionale dalle altre funzioni.

Convenzione di notazione:
- `close_t` = prezzo di chiusura al tempo `t`;
- `MA_k(X)` = media mobile semplice a `k` periodi della serie `X`;
- `STD_k(X)` = deviazione standard a `k` periodi della serie `X`;
- `trd` = `context['target_return_days']` (default 5);
- `vol_window` = `context['volatility_window']` (default 20, non usato direttamente
  nelle formule: le finestre sono hardcoded 5/10/20/50/60).

## 1. Time features — `add_time_features(df)`

| Feature | Formula | Note |
|---|---|---|
| `day_of_week` | `date.dayofweek` | 0=Monday, 6=Sunday |
| `time_since_high` | giorni trascorsi dall'ultimo massimo storico di `close` | Iterazione riga per riga; 0 il giorno del nuovo massimo |

Ordine di costruzione:
1. `timestamp = to_datetime(date)` (colonna temporanea).
2. `date` diventa l'indice.
3. `day_of_week`.
4. `time_since_high` iterando le righe: se `close > max_close` aggiorna il massimo e
   azzera i giorni; altrimenti incrementa.
5. Rimozione di `timestamp`, ripristino di `date` come colonna.

## 2. Financial features — `add_financial_features(df)`

Tutte le finestre sono calcolate su `close` e `volume`.

### 2.1 Returns

- `log_return = ln(close_t / close_{t-trd})`

### 2.2 Momentum

- `mom_5  = close.pct_change(5)`
- `mom_10 = close.pct_change(10)`
- `mom_20 = close.pct_change(20)`

### 2.3 Volatility

- `vol_10 = STD_10(log_return)`
- `vol_20 = STD_20(log_return)`
- `vol_ratio = vol_10 / vol_20`

### 2.4 Mean reversion

- `ma_20 = MA_20(close)`
- `std_20 = STD_20(close)`
- `zscore_20 = (close - ma_20) / std_20`

### 2.5 Trend strength

- `ma_50 = MA_50(close)`
- `trend_50 = (close - ma_50) / ma_50`

### 2.6 Volume signal

- `vol_mean_20 = MA_20(volume)`
- `vol_std_20 = STD_20(volume)`
- `volume_zscore = (volume - vol_mean_20) / vol_std_20`

### 2.7 Intraday range

- `hl_range = (high - low) / close`

### 2.8 Close position in range

- `close_range = (close - low) / (high - low)`

### 2.9 Momentum volatility-adjusted

- `mom_vol_adj = mom_20 / vol_20`

### 2.10 Target

- `target_raw = log_return.shift(-trd) / vol_20`

> `shift(-trd)` porta in avanti il rendimento futuro `t+trd`, quindi il target è il
> **rendimento logaritmico futuro a `trd` giorni normalizzato per la volatilità**.

- `target = clip(target_raw, clip_min, clip_max)`
  con `clip_min`, `clip_max` dal `conf.ini` del **modello** (sezione `[predictions]`).

### 2.11 Pulizia

- `dropna` su: `log_return, mom_5, mom_10, mom_20, vol_10, vol_20, vol_ratio,
  zscore_20, trend_50, volume_zscore, mom_vol_adj` (le righe con questi NaN vengono
  rimosse; nota: `hl_range`, `close_range`, `volume_shock`, `vol_regime`, `mom_rank`
  **non** sono nella lista dei dropna).

## 3. Feature cross-sezionali (su tutto il dataset aggregato)

Applicate dopo `pd.concat` di tutti i simboli.

### 3.1 `cross_sectional_momentum_rank(df)` → `mom_rank`

- `mom_20 = close.pct_change(20)` raggruppato per `symbol`.
- Per ogni data: `mean = mean(mom_20)`, `std = std(mom_20)` **a livello di
  portafoglio** (transform per `date`).
- `mom_rank = (mom_20 - mean) / std` → z-score cross-sezionale del momentum.

### 3.2 `volume_shock_feature(df)` → `volume_shock`

- `vol_ma = MA_20(volume)` raggruppato per `symbol`.
- `volume_shock = volume / vol_ma`

### 3.3 `volatility_regime(df)` → `vol_regime`

- `log_return_1 = ln(close / close.shift(1))`
- `vol_10 = STD_10(log_return_1)` per `symbol`.
- `vol_60 = STD_60(log_return_1)` per `symbol`.
- `vol_regime = vol_10 / vol_60`

## 4. Feature esterne — `add_forex_features` / `add_index_features`

- `get_forex()`: chiuse giornaliere di coppie forex da yfinance
  (es. `GBPUSD=X, EURUSD=X, USDJPY=X, USDCAD=X, USDCHF=X, AUDUSD=X, NZDUSD=X,
  GC=F, BZ=F, CNYUSD=X`). Configurabili in `[features] forex` del conf.ini del modello.
  Nome colonna: parte prima di `=` (es. `GBPUSD`).
- `get_index()`: chiuse giornaliere degli indici
  (es. `^GSPC, ^DJI, ^IXIC, ^RUT, ^VIX1D`). Configurabili in `[features] indexes`.
  Nome colonna: simbolo yfinance (es. `^GSPC`).
- Merge `left` su `date`.

> Questi dati vengono sempre scaricati da **yfinance**, indipendentemente dal
> `data_source` del portfolio.

## 5. Feature categoriche — `add_country(df)`

1. Legge `portfolio.csv` del **modello** (definisce le categorie di riferimento).
   Filtra righe con `Country`/`Sector` lunghi > 1.
2. `list_countries` = valori unici `Country` del modello → `CategoricalDtype`.
3. Legge `portfolio.csv` del **portfolio di trading**, crea mappa
   `Symbol → Country` e fa merge `left` su `symbol`.
4. `Country` → `CategoricalDtype(categories=list_countries)` → `pd.get_dummies(...,
   prefix='country', dtype=int)`.
5. Idem per `Sector` (categorie dal modello, mappa dal portfolio) →
   `pd.get_dummies(..., prefix='sector', dtype=int)`.
6. Elimina la colonna `Symbol`.
7. L'**Industry** è commentata/rimossa (non usata come dummy).

> Nota: le categorie sono definite dal portfolio del **modello**, quindi se il
> portfolio di trading contiene un Paese/Settore non presente nel modello, la dummy
> genererà una colonna in più (o un NaN se convertito a category con categorie
> fisse). Le categorie fisse (`CategoricalDtype`) fanno sì che `get_dummies` generi
> **tutte** le colonne del modello anche se assenti nei dati, mantenendo l'allineamento
> tra training e inference.

## 6. Outlier clipping — `clip_outliers(df, columns)`

Per ogni colonna nella lista, clippa ai quantili 1% e 99%:

- `lower = df[col].quantile(0.01)`
- `upper = df[col].quantile(0.99)`
- `df[col] = df[col].clip(lower, upper)`

Colonne clippate in `add_features_portfolio`:

`log_return, mom_5, mom_10, mom_20, mom_rank, vol_10, vol_20, vol_ratio, vol_regime,
volume_shock, zscore_20, trend_50, volume_zscore, mom_vol_adj, hl_range, close_range`
(nota: `zscore_20` è elencato due volte, effetto identico).

## 7. Riordino e pulizia — `reorder_columns(df)` + drop

- Rimuove la colonna `volume` (giudicata "cattiva per il modello").
- Ordina le colonne in ordine alfabetico.
- In `add_features_portfolio`, dopo tutte le trasformazioni, elimina anche
  `close`, `low`, `high`, `open` (i prezzi grezzi non servono al modello).

> **Importante per la riscrittura:** il modello riceve **solo** le feature, non i
> prezzi grezzi né il volume. La previsione usa quindi esclusivamente le colonne
> ordinate alfabeticamente di f001 (escluse `date`, `symbol`, `target`).

## 8. Ordine di esecuzione in `add_features_portfolio()`

1. Per ogni simbolo: `add_time_features` → `add_financial_features`; concatena.
2. `cross_sectional_momentum_rank` → `mom_rank`.
3. `volume_shock_feature` → `volume_shock`.
4. `volatility_regime` → `vol_regime`.
5. `add_forex_features`.
6. `add_index_features`.
7. `clip_outliers`.
8. `add_country`.
9. `reorder_columns`.
10. Drop di `close, low, high, open`.
11. Salvataggio in `f001_features.parquet`.

## 9. Osservazioni per la riscrittura

- Le finestre (5/10/20/50/60) sono **hardcoded** nei nomi colonna e nelle funzioni;
  solo `target_return_days` e `volatility_window` sono configurabili (e quest'ultimo
  non viene usato nelle formule effettive).
- `hl_range` e `close_range` richiedono `high`/`low`, non disponibili con Alpaca.
- Le feature cross-sezionali e le dummies country/sector richiedono l'intero
  dataset del giorno, non una singola azione.
- Il target dipende da `vol_20` e da `log_return.shift(-trd)`: richiede che ogni
  simbolo abbia abbastanza storia per calcolare la finestra di volatilità e il
  look-ahead del target.
