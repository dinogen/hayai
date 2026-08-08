# 07 — Integrazioni esterne

Questo documento descrive le tre integrazioni esterne dell'applicazione:
**Yahoo Finance** (yfinance), **Alpaca** (alpaca-py) e **Telegram** (telethon).

## 1. Yahoo Finance — `yfinance`

Usato sempre per: storico prezzi (in modalità `yfinance`), ultimi prezzi, forex,
indici, e come back-end della modalità "simulata".

### 1.1 Download storico — `fetch_quotes_yfinance`

- API: `yf.download(symbol, period="5y", interval="1d")`.
- Output: dataframe multi-colonna che viene rinominato in
  `symbol, date, open, close, high, low, volume`.
- `date` deriva dall'indice `Date`.

### 1.2 Ultimo prezzo — `get_latest_price_yfinance`

- API: `yf.Ticker(symbol)` + `ticker.info`.
- Logica: se `'postMarketPrice'` è in `ticker.info` → usa quello, altrimenti
  `'currentPrice'`.
- Il simbolo `MYCASH` viene saltato.
- Nota: fare `ticker.info` per **ogni** simbolo è lento (chiamata HTTP per simbolo);
  per portafogli di ~1000 simboli è un collo di bottiglia.

### 1.3 Forex — `get_forex`

- `yf.download(symbols, period="5y", interval="1d")`.
- Prende la colonna `Close`; rinomina le colonne togliendo il suffisso `=...`
  (`GBPUSD=X` → `GBPUSD`).
- Aggiunge `date` (`pd.to_datetime(index).date`).
- Simboli da `context['forex']` (conf.ini del modello).

### 1.4 Indici — `get_index`

- `yf.download(symbols, period="5y", interval="1d")`.
- Prende la colonna `Close`; i nomi colonna restano i simboli yfinance
  (es. `^GSPC`, `FTSEMIB.MI`).
- Aggiunge `date`.
- Simboli da `context['indexes']` (conf.ini del modello).

### 1.5 Modalità "simulata" (yfinance)

In `hayai_trade.py`, con `data_source=yfinance`, l'acquisto non tocca Alpaca ma
aggiorna i file parquet locali (`position_new_qty.parquet`, `actual_positions.parquet`).

> Limite noto: la simulazione gestisce **solo acquisti** (`_place_order_buy_yfinance`).
> Le vendite e gli short chiamano comunque Alpaca (vedi `05` e `09`).

## 2. Alpaca — `alpaca-py`

Usato per: storico prezzi (modalità `alpaca`), ultimo prezzo, posizioni reali,
equity reale, esecuzione ordini (paper).

### 2.1 Client

- `TradingClient(api_key, secret_key, paper=True)` — trading (paper account).
- `StockHistoricalDataClient(api_key, secret_key)` — dati storici e ultimi trade.
- Le chiavi vengono dal `secret.ini` del portfolio (sezione `[portfolio]`).

### 2.2 Storico — `fetch_quotes_alpaca`

- `StockBarsRequest(symbol_or_symbols=[symbol], timeframe=TimeFrame.Day,
  start=today-5y, end=today-1day)`.
- `client.get_stock_bars(request_params)`.
- Output: `symbol, date, close, volume` (da `timestamp` → `date`).
- **Mancano** `open/high/low` (vedi `02`, `09`).

### 2.3 Ultimo prezzo — `get_latest_price_alpaca`

- `StockLatestTradeRequest(symbol_or_symbols=symbols)`.
- `client.get_stock_latest_trade(request_params)`.
- `prices = {k: v.price for k, v in result.items()}` → dataframe `symbol, price`.

### 2.4 Posizioni reali — `get_actual_position_alpaca`

- `client.get_all_positions()`.
- Converte in dataframe, `qty` a float, seleziona `symbol, qty` → rinominato
  `symbol, qty_old`.

### 2.5 Equity reale — `get_equity_alpaca`

- `client.get_account()` → `account.equity` (float).
- È la fonte di equity **affidabile** (con yfinance è stimata dai parquet).

### 2.6 Ordini — `hayai_trade.py`

- `MarketOrderRequest(symbol, qty, side=BUY/SELL, time_in_force=DAY)`.
- `tc.submit_order(order_data)`.
- `client.close_position(symbol)` per azzerare una posizione.
- Gli errori `APIError` vengono loggati e **non** interrompono il loop
  (gli ordini successivi continuano).

### 2.7 Nota sui prezzi (da `prova.ipynb`)

- Il notebook `prova.ipynb` annota: "Prezzo quando compro: ask, Prezzo quando vendo:
  bid, Usare la media dei due" — nota di progettazione non implementata nel codice
  (i prezzi usati sono quelli di `get_latest_price`).

## 3. Telegram — `telethon`

Modulo `hayai_msg.py`.

- `send_message(msg)`: `client.send_message(chat_id, msg)`.
- `send_file(file_path, caption)`: `client.send_file(chat_id, file_path,
  caption=caption)`.
- Uso: `TelegramClient('bot', api_id, api_hash).start(bot_token=bot_token)`.
- Credenziali dal contesto (`telegram_api_id`, `telegram_api_hash`,
  `telegram_bot_token`, `telegram_chat_id`).

### Quando vengono inviate notifiche

| Fase | Notifica |
|---|---|
| `-s` (build signals) | File di log via `send_file` |
| `-r` (report) | Report HTML via `send_file` (solo se `create_report` restituisce True) |
| `-e` (execute trades) | Messaggio di testo "HAYAI has executed trades..." |

## 4. Dipendenze (requirements.txt)

Il file `requirements.txt` (codificato UTF-16) elenca pin espliciti. Librerie chiave:

- `alpaca-py==0.43.2`: trading + dati Alpaca.
- `pandas==3.0.1`, `numpy==2.4.2`, `pyarrow==23.0.1`: dati e parquet.
- `keras==3.13.2`: modello (funziona con il backend).
- `torch==2.10.0`, `tensorboard==2.20.0`: backend/visualizzazione.
- `scikit-learn==1.8.0`: `train_test_split` nei notebook.
- `matplotlib==3.10.8`, `seaborn==0.13.2`: EDA nei notebook.
- `ipykernel`, `jupyter_client`: notebook.

> **Attenzione:** `yfinance` e `telethon` **non** sono in `requirements.txt` pur
> essendo importati dal codice (`import yfinance as yf`, `from telethon import
> TelegramClient`). Il file è quindi incompleto e non installabile da zero senza
> integrare queste dipendenze.

## 5. Comportamenti asimmetrici yfinance vs alpaca (tabella di sintesi)

| Operazione | yfinance | alpaca |
|---|---|---|
| Storico OHLCV | OHLCV completo | solo `close, volume` |
| Ultimo prezzo | `Ticker.info` (lento, per-simbolo) | batch `get_stock_latest_trade` (veloce) |
| Posizione attuale | da `f008_actual.parquet` (tutti i simboli + MYCASH) | da Alpaca (solo detenuti, no MYCASH) |
| Equity | stimata dai parquet (logica discutibile) | `account.equity` reale |
| Esecuzione ordini | simulata (solo buy) | reale su paper account |
| Forex / indici | sempre yfinance | sempre yfinance (non Alpaca) |

> Le differenze nella posizione attuale e nell'equity hanno impatto diretto sulla
> correttezza del calcolo delle nuove posizioni (vedi `09`).

## 6. Considerazioni per la riscrittura

- Alpaca non fornisce `high/low` per lo storico: o si adottano solo feature
  compatibili (senza `hl_range`, `close_range`), o si usa yfinance anche in
  produzione, o si integra un provider che fornisca OHLCV completo.
- `yf.Ticker.info` è troppo lento per portafogli grandi: da sostituire con batch
  (es. `download` di ticker multipli) o altro provider.
- Telegram è un canale di notifica "fire-and-forget": nel nuovo sistema meglio
  un sistema di notifiche disaccoppiato (eventi → canali multipli).
- Gli errori API (Alpaca) vengono solo loggati e non gestiti: il nuovo sistema deve
  definire retry, dead-letter e stato degli ordini.
