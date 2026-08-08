# 02 — Pipeline dati

Questo documento descrive l'**ingestion** dei dati e lo **schema di ogni file** prodotto
dall'applicazione. I file intermedi sono parquet salvati tramite pandas.

## 1. Struttura delle cartelle

```
data/
├── <portfolio_id>/                 # portfolio di trading
│   ├── conf.ini                    # configurazione del portfolio
│   ├── secret.ini                  # chiavi Alpaca del portfolio
│   ├── portfolio.csv               # lista simboli (download da Nasdaq screener)
│   ├── hist/                       # dati storici grezzi, un parquet per simbolo
│   ├── f001_features.parquet
│   ├── f002_predictions.parquet
│   ├── f003_weights.parquet
│   ├── f005_position_new.parquet
│   ├── f006_position_new_qty.parquet
│   ├── f007_orders.parquet
│   ├── f008_actual.parquet
│   └── report_{portfolio_id}.html
└── <model_id>/                     # portfolio modello (id che inizia con "model_")
    ├── conf.ini                    # parametri modello + forex/indexes
    ├── secret.ini
    ├── portfolio.csv               # universo di addestramento (ampio, ~1000 symbol)
    ├── large_portfolio.csv         # (opzionale) versione più ampia
    ├── model.keras                 # modello addestrato
    ├── model_summary.txt           # summary del modello
    ├── mins.csv / maxs.csv         # parametri di normalizzazione min-max
    └── hist/                       # storico dei simboli del modello
```

## 2. Dati di input

### 2.1 `portfolio.csv`

Scaricato da https://www.nasdaq.com/market-activity/stocks/screener.

Colonne originali (Nasdaq):

| Colonna | Esempio | Note |
|---|---|---|
| `Symbol` | `AAOI` | Ticker |
| `Name` | `Applied Optoelectronics Inc.` | Nome società |
| `Last Sale` | `$44.30` | Ultimo prezzo |
| `Net Change` | `6.17` | Variazione |
| `% Change` | `16.181%` | Variazione % |
| `Market Cap` | `3024733873.00` | Capitalizzazione |
| `Country` | `United States` | Paese |
| `IPO Year` | `2013` | Anno IPO |
| `Volume` | `5410388` | Volume |
| `Sector` | `Technology` | Settore |
| `Industry` | `Semiconductors` | Industria |

Regole di pulizia applicate:

- In `create_context` si leggono solo le righe con `Country != ''` e `Sector != ''`
  (`keep_default_na=False`, quindi si confronta con stringa vuota).
- `add_country` filtra anche le righe con `Country` o `Sector` di lunghezza ≤ 1.
- Le colonne usate dalla logica sono: `Symbol`, `Country`, `Sector` (per le dummies)
  e `Market Cap` (per il campionamento del portfolio modello).

### 2.2 `secret.ini` (globale, in root del progetto)

Contiene le credenziali Telegram:

```ini
[telegram]
api_id = ...
api_hash = ...
bot_token = ...
chat_id = ...
```

### 2.3 `secret.ini` (per portfolio)

Contiene le chiavi Alpaca:

```ini
[portfolio]
api_key = ...
secret_key = ...
```

### 2.4 `conf.ini` (per portfolio e per modello)

Schema completo in `06-configurazione.md`.

## 3. Fase di ingestione (`-i`)

Funzioni: `hayai_dao.fetch_quotes_portfolio()` → `hayai_bo.add_features_portfolio()`.

### 3.1 `fetch_quotes_portfolio()`

Per ogni simbolo in `context['symbols']`:

1. Se esiste già `hist/{symbol}.parquet`:
   - calcola `max(mtime, atime, ctime)` del file;
   - se `now - max(...) < 4h` → **salta** il download (cache).
2. Altrimenti scarica con `fetch_quotes(symbol, client)` (yfinance o alpaca).
3. Se il dataframe ha **meno di 365 righe** → scarta il simbolo (non salvato).
4. Salva `hist/{symbol}.parquet`.

### 3.2 `fetch_quotes_yfinance(symbol)`

- `yf.download(symbol, period="5y", interval="1d")`.
- Colonne finali: `symbol, date, open, close, high, low, volume`.

### 3.3 `fetch_quotes_alpaca(symbol, client)`

- `StockBarsRequest`, timeframe `Day`, da 5 anni fa a ieri.
- Colonne finali: `symbol, date, close, volume` (mancano `open/high/low`).

> **Differenza critica:** yfinance restituisce anche `open/high/low`, Alpaca solo
> `close/volume`. Tuttavia `add_financial_features` usa `high` e `low` per
> `hl_range` e `close_range`. In modalità Alpaca queste feature saranno `NaN`.
> Vedi `09-vincoli-riscrittura.md`.

### 3.4 `add_features_portfolio()`

Aggrega tutti gli `hist/*.parquet`, applica il feature engineering (vedi `03`) e
salva `f001_features.parquet`. Al termine elimina le colonne `close`, `low`, `high`,
`open` (la colonna `volume` è già rimossa da `reorder_columns`).

## 4. File intermedi (parquet)

### 4.1 `hist/{SYMBOL}.parquet`

| Colonna | Tipo | Note |
|---|---|---|
| `symbol` | str | Ticker |
| `date` | date | Data della barra (per Alpaca deriva da `timestamp`) |
| `open` | float | Solo yfinance |
| `close` | float | Prezzo di chiusura |
| `high` | float | Solo yfinance |
| `low` | float | Solo yfinance |
| `volume` | int | Volume |

### 4.2 `f001_features.parquet` (`FILE_FEATURES`)

| Colonna | Tipo | Significato |
|---|---|---|
| `date` | date | Data della barra |
| `symbol` | str | Ticker |
| `day_of_week` | int | 0=Lunedì ... 6=Domenica |
| `time_since_high` | int | Giorni dall'ultimo massimo di `close` |
| `log_return` | float | `ln(close / close.shift(trd))`, trd = `target_return_days` |
| `mom_5`, `mom_10`, `mom_20` | float | `pct_change(5/10/20)` di `close` |
| `vol_10`, `vol_20` | float | std di `log_return` su 10/20 giorni |
| `vol_ratio` | float | `vol_10 / vol_20` |
| `zscore_20` | float | `(close - MA20) / STD20` |
| `trend_50` | float | `(close - MA50) / MA50` |
| `volume_zscore` | float | `(volume - MA20_vol) / STD20_vol` |
| `hl_range` | float | `(high - low) / close` |
| `close_range` | float | `(close - low) / (high - low)` |
| `mom_vol_adj` | float | `mom_20 / vol_20` |
| `mom_rank` | float | z-score cross-sezionale di `mom_20` per data |
| `volume_shock` | float | `volume / MA20(volume)` |
| `vol_regime` | float | `vol_10 / vol_60` |
| `country_*` | int (0/1) | Dummies del Paese |
| `sector_*` | int (0/1) | Dummies del Settore |
| `forex_*` | float | Chiuse forex (merge su data) |
| `index_*` | float | Chiuse indici (merge su data) |
| `target` | float | `clip(log_return.shift(-trd)/vol_20, clip_min, clip_max)` |

- Una riga per **symbol per data**.
- La colonna `volume` viene rimossa (cattiva per il modello).
- Dopo il feature engineering vengono rimosse anche `close`, `low`, `high`, `open`.

### 4.3 `f002_predictions.parquet` (`FILE_PREDICTIONS`)

| Colonna | Significato |
|---|---|
| (tutte le colonne di f001 tranne `date`, `symbol`, `target`) | features normalizzate |
| `prediction` | output del modello denormalizzato e clippato |
| `symbol` | Ticker (ripristinato) |
| `date` | Data (ripristinata) |

- **Solo l'ultima data valida** del dataset (`df['date'] == df["date"].max()`).
- Una riga per symbol.

### 4.4 `f003_weights.parquet` (`FILE_WEIGHTS`)

| Colonna | Significato |
|---|---|
| `symbol` | Ticker |
| `prediction` | Predizione del modello |
| `vol_20` | Volatilità 20 giorni |
| `weight` | Peso normalizzato (`prediction.clip / vol_20`, normalizzato a somma abs = 1) |

- Solo `n_long` simboli long (peso > 0, più alti) e `n_short` simboli short
  (peso < 0, più negativi).
- Molti simboli del portfolio hanno peso 0 (non compaiono nel file).

### 4.5 `f005_position_new.parquet` (`FILE_POSITION_NEW`)

| Colonna | Significato |
|---|---|
| `symbol` | Ticker |
| `qty_old` | Quantità attuale (da posizione reale o da f008) |
| `value_old` | Valore attuale (solo yfinance) |
| `weight_new` | Nuovo peso obiettivo |

- Una riga per simbolo del portfolio (merge posizione attuale con pesi; `.fillna(0)`).

### 4.6 `f006_position_new_qty.parquet` (`FILE_POSITION_NEW_QTY`)

| Colonna | Significato |
|---|---|
| `symbol` | Ticker (incluso `MYCASH`) |
| `qty_old` | Quantità attuale |
| `weight_new` | Peso obiettivo |
| `price` | Ultimo prezzo (`get_latest_price`) |
| `value_new` | `qty_new * price` |
| `qty_new` | Quantità obiettivo arrotondata |
| `qty_diff` | `qty_new - qty_old` (post soglia minima) |
| `qty_diff_perc` | `qty_diff / (qty_old se ≠ 0 altrimenti qty_new)` |

- La riga `MYCASH` viene estratta, elaborata a parte (contiene `cash_flow`) e
  riaggiunta alla fine.
- Il file finale ha colonne: `symbol, qty_old, qty_new, qty_diff, price, weight_new,
  value_new`.

### 4.7 `f007_orders.parquet` (`FILE_ORDERS`)

| Colonna | Significato |
|---|---|
| `symbol` | Ticker |
| `operation` | `BUY` / `SELL` / `CLOSE` |
| `qty` | Quantità: positiva per `BUY`/`SELL`; per `CLOSE` mantiene il segno di `qty_old` (positiva se si chiude una long, **negativa** se si chiude una short) |
| `price` | Prezzo di esecuzione stimato |

- Non contiene `MYCASH`.
- Regole di generazione: vedi `05-logica-trading.md` (tabella delle transizioni).

### 4.8 `f008_actual.parquet` (`FILE_ACTUAL`)

| Colonna | Significato |
|---|---|
| `date` | Data della posizione |
| `symbol` | Ticker (incluso `MYCASH`) |
| `qty` | Quantità |
| `price` | Prezzo |
| `value` | `qty * price` |

- Contiene la **storia delle posizioni per data**.
- La prima riga è `MYCASH` con `qty=1` e `price` = capitale iniziale.
- `update_actual_position` sostituisce la riga della data odierna se già presente.

### 4.9 Report HTML (`report_{portfolio_id}.html`)

Generato da `util.create_report(df_new_qty)`:

- Tabella con simboli (escluso `MYCASH`), quantità, prezzo, valore.
- Riepilogo: Total Long, Total Short, Net Position, Cash, Total Portfolio Value.
- Usa solo le righe con `qty_new != 0`.

## 5. Convenzioni di calcolo

- **Peso**: `prediction.clip(clip_min, clip_max) / vol_20`, poi normalizzato a
  somma dei valori assoluti = 1.
- **Equity**: `value_old(MYCASH) + Σ value_new(simboli)` nella fase di definizione
  quantità; `equity * risk_percentage` è il capitale investito.
- **Cash flow**: `-qty_diff * price` per ogni simbolo; il totale viene sommato a
  `MYCASH.value_old` per ottenere il nuovo `value_new` della cassa.

## 6. Nomi file hardcoded (potenziali problemi per la riscrittura)

Diversi punti usano stringhe hardcoded invece delle costanti `FILE_*`:

| Punto | Stringa hardcoded | Costante corrispondente |
|---|---|---|
| `hayai_bo.execution()` | `"position_new_qty.parquet"` | `FILE_POSITION_NEW_QTY` (f006) |
| `hayai_trade._place_order_buy_yfinance` | `"position_new_qty.parquet"` | `FILE_POSITION_NEW_QTY` |
| `hayai_trade._place_order_buy_yfinance` | `"actual_positions.parquet"` | `FILE_ACTUAL` (f008) |
| `hayai_dao.get_equity_yfinance` | `"actual_positions.parquet"` | `FILE_ACTUAL` |

> Nota: i nomi `actual_positions.parquet` (senza prefisso f0NN) sono **diversi** da
> `f008_actual.parquet`. Nella cartella dati esistono entrambe le versioni: i file
> rinominati con prefisso `f0NN` (`f001_features.parquet`, `f008_actual.parquet`, ...)
> e file "legacy" con nomi storici (`features.parquet`, `predictions.parquet`,
> `weights.parquet`, `position_new.parquet`, `position_new_qty.parquet`,
> `positions.parquet`). `actual_positions.parquet` è un nome usato solo in codice
> (nella modalità simulata yfinance) ma non appare nella cartella attuale: è un
> residuo non allineato con la convenzione `f0NN` e rappresenta un rischio di
> divergenza tra file atteso e file prodotto.
