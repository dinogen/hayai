# 05 — Logica di trading

Questo documento descrive la logica che trasforma le **predizioni** in **pesi**,
**posizioni**, **quantità**, **ordini** e **trade eseguiti**. Tutto in `hayai_bo.py`
(più `hayai_trade.py` per l'esecuzione).

## 1. Definizione dei pesi — `define_weight()`

Input: `f002_predictions.parquet`. Output: `f003_weights.parquet`.

Passi:

1. Prende `symbol`, `prediction`, `vol_20`.
2. `weight = prediction.clip(clip_min, clip_max) / vol_20`.
3. Ordina per `weight` **decrescente**.
4. Seleziona:
   - `df_long` = le prime `n_long` righe con `weight > 0`;
   - `df_short` = le ultime `n_short` righe con `weight < 0`.
5. Concatena: `df = df_long + df_short`.
6. **Normalizza**: `weight = weight / sum(|weight|)`.
7. **Asserzione**: `0.99 < sum(|weight|) < 1.01`.

Risultato: solo `n_long` asset long e `n_short` asset short con pesi la cui somma
dei valori assoluti è 1. Gli altri asset non compaiono (peso implicito 0).

> Interpretazione: il peso di un asset è proporzionale alla predizione (rendimento
> atteso normalizzato) diviso la volatilità (inverso del rischio). Long = predizione
> positiva, short = predizione negativa.

## 2. Costruzione nuova posizione — `build_new_position()`

Input: `f003_weights.parquet` + posizione attuale. Output: `f005_position_new.parquet`.

```
df_new = weights[['symbol', 'weight']]  → rinominata ['symbol', 'weight_new']
df_old = dao.get_actual_position()      # ['symbol','qty_old',('value_old')]
df = merge(df_old, df_new, on='symbol', how='left').fillna(0)
```

- `qty_old` = quantità attuale (da Alpaca reale o da `f008_actual.parquet`).
- `value_old` = valore attuale (presente solo con yfinance).
- I simboli presenti in `df_old` ma non in `df_new` ricevono `weight_new = 0`.

> **Discrepanza (vedi `09`):** con `data_source=alpaca`, `get_actual_position_alpaca`
> restituisce solo i simboli effettivamente detenuti e **non** la riga `MYCASH`.
> Il merge `left` su `df_old` perde quindi gli asset con peso nuovo ma senza
> posizione attuale (non ancora detenuti), impedendo di generare ordini di apertura.
> Con yfinance `df_old` contiene tutti i simboli del portfolio (inizializzati a 0).

## 3. Definizione delle quantità — `define_new_quantity()`

Input: `f005_position_new.parquet` + prezzi. Output: `f006_position_new_qty.parquet`.

Passi (pseudocodice fedele):

```
row_mycash = df[df.symbol == MYCASH]        # estratta e gestita a parte
df = df[df.symbol != MYCASH]
df_price = get_latest_price(df.symbol)
df = merge(df, df_price, on='symbol', how='outer').fillna(0)

value_new = qty_old * price                 # valore attuale (prezzo corrente)
equity    = row_mycash.value_old + sum(value_new)     # equity stimata
capital   = equity * risk_percentage        # capitale investito

value_new   = weight_new * capital
qty_new     = round(value_new / price)
qty_diff    = qty_new - qty_old
denominators = where(qty_old != 0, qty_old, qty_new)
qty_diff_perc = qty_diff / denominators
qty_diff_perc = fillna(0)

# se la variazione è trascurabile, si azzera il diff
where(|qty_diff_perc| < qty_diff_perc_min): qty_diff = 0

qty_new   = qty_old + qty_diff              # ricalcolo
value_new = qty_new * price
cash_flow = -qty_diff * price               # flusso di cassa per asset

total_cash_flow = sum(cash_flow)
row_mycash.value_new = row_mycash.value_old + total_cash_flow
row_mycash: price=1, qty_new=1, qty_diff=0, qty_diff_perc=0

df = concat(df, row_mycash)
df = df[['symbol','qty_old','qty_new','qty_diff','price','weight_new','value_new']]
```

Concetti chiave:

- **Equity** = valore corrente di cassa + valore corrente delle posizioni.
- **Capitale investito** = `equity * risk_percentage` (default 0.8 → il 20% resta
  in cassa).
- `value_new` obiettivo = `weight_new * capital`; `qty_new = round(value_new/price)`.
- **Soglia minima di trade**: se la variazione percentuale è inferiore a
  `qty_diff_perc_min` (default 0.2 = 20%), `qty_diff` viene azzerato → nessun trade.
- `cash_flow` è negativo per acquisti, positivo per vendite.

## 4. Definizione degli ordini — `define_orders()`

Input: `f006_position_new_qty.parquet` (escluso `MYCASH`). Output:
`f007_orders.parquet` con colonne `symbol, operation, qty, price`.

Logica per ogni asset (in base a `qty_old`, `qty_new`, `qty_diff`):

| Transizione | `qty_old` | `qty_new` | Ordini generati |
|---|---|---|---|
| Nessuna variazione | qualsiasi | `qty_diff == 0` | *(nessun ordine)* |
| Apertura long | 0 | > 0 | `BUY qty_new` |
| Apertura short | 0 | < 0 | `SELL abs(qty_new)` |
| Chiusura long | > 0 | 0 | `CLOSE qty_old` |
| Chiusura short | < 0 | 0 | `CLOSE qty_old` |
| Long → più long | > 0 | > 0, `qty_old < qty_new` | `BUY abs(qty_diff)` |
| Long → meno long | > 0 | > 0, `qty_old > qty_new` | `SELL abs(qty_diff)` |
| Short → più short | < 0 | < 0, `qty_old > qty_new` (es. -100 → -150) | `SELL abs(qty_diff)` |
| Short → meno short | < 0 | < 0, `qty_old < qty_new` (es. -150 → -100) | `BUY abs(qty_diff)` |
| **Inversione** long → short | > 0 | < 0 | `CLOSE qty_old` + `SELL abs(qty_new)` |
| **Inversione** short → long | < 0 | > 0 | `CLOSE qty_old` + `BUY qty_new` |

Note:

- **Segno di `qty` (incoerenza nel codice):** `BUY` e `SELL` usano `abs(qty)` (quindi
  positive); `CLOSE` usa `qty_old` **così com'è**: per la chiusura di una posizione
  long è positiva, per la chiusura di una posizione short è **negativa**
  (`hayai_bo.py:393-394, 409`). I consumatori del file devono gestire questa
  asimmetria.
- `price` è il prezzo corrente (non il prezzo di esecuzione reale).
- L'ordine degli ordini nel file è l'ordine di iterazione del dataframe: la sequenza
  è influente (prima si chiude, poi si apre).

> Il `Readme.md` descriveva un formato `date, n, symbol, qty, price` con `qty` firmata
> (+ per buy, − per sell): quel formato è **superato**. Il codice attuale produce
> `symbol, operation, qty, price`.

## 5. Aggiornamento posizione attuale — `update_actual_position()`

Input: `f006_position_new_qty.parquet`. Output (aggiornamento): `f008_actual.parquet`.

1. Seleziona `symbol, qty_new, price, value_new` → rinomina in `symbol, qty, price,
   value`; aggiunge `date = oggi`.
2. Legge `f008_actual.parquet` (storico per data).
3. **Sostituisce** la riga della data odierna se esiste (`df_actual = df_actual[
   df_actual.date != today]`), poi concatena la nuova posizione.
4. Ordina per `date` e salva.

## 6. Esecuzione dei trade — `execution()`

Input: `position_new_qty.parquet` (**hardcoded**, non la costante `FILE_...`).
Output: ordini su Alpaca (o simulazione yfinance).

Logica per ogni asset (identica nei concetti a `define_orders` ma con chiamate reali):

- `qty_new == 0` → `client.close_position(symbol)` (chiude qualsiasi posizione).
- `qty_old == 0, qty_new > 0` → `place_order_buy(qty_new)`.
- `qty_old == 0, qty_new < 0` → `place_order_short(abs(qty_new))`.
- `qty_old > 0` (long esistente):
  - `qty_new > 0, qty_diff > 0` → buy `qty_diff`;
  - `qty_new > 0, qty_diff < 0` → sell `abs(qty_diff)`;
  - `qty_new < 0` → `close_position` + short `abs(qty_new)`.
- `qty_old < 0` (short esistente):
  - `qty_new < 0, qty_diff > 0` → buy `abs(qty_diff)`;
  - `qty_new < 0, qty_diff < 0` → short `abs(qty_diff)`;
  - `qty_new > 0` → `close_position` + buy `qty_new`.

Dettagli esecuzione (da `hayai_trade.py`):

- `place_order_buy(symbol, qty, tc)`:
  - **alpaca** → `_place_order_buy_alpaca`: `MarketOrderRequest`, `side=BUY`,
    `time_in_force=DAY`, via `tc.submit_order` (errore `APIError` loggato).
  - **yfinance** → `_place_order_buy_yfinance`: simula l'acquisto modificando i file
    `position_new_qty.parquet` e `actual_positions.parquet` (aggiorna `qty`, `value`
    e la cassa `MYCASH`).
- `place_order_sell(tc, symbol, qty)`: solo Alpaca (market order DAY, `side=SELL`);
  se `qty < 1` dopo il round, logga e salta.
- `place_order_short(tc, symbol, qty)`: solo Alpaca (market order DAY, `side=SELL`).

> **Discrepanza (vedi `09`):** in modalità yfinance solo l'**acquisto** è simulato;
> le vendite/short (`place_order_sell`, `place_order_short`) invocano comunque
> `tc.submit_order` su Alpaca anche se `data_source=yfinance` (il parametro `tc`
> potrebbe non essere passato correttamente in `execution`). Da verificare.

> **Bug argomenti `place_order_buy`:** la firma è `place_order_buy(symbol, qty, tc)`
> ma `execution()` chiama `trade.place_order_buy(client, symbol, qty_new)`
> (`hayai_bo.py:454`): gli argomenti sono **scambiati** (`symbol`=client,
> `qty`=symbol, `tc`=qty_new). Con `data_source=alpaca` questo causa un crash
> (`assert qty > 0` con `qty` stringa → TypeError) al primo ordine buy. Le chiamate
> a `place_order_sell` e `place_order_short` invece sono corrette
> (`(tc, symbol, qty)`).

## 7. Inizializzazione portfolio — `init_portfolio(initial_amount)`

Scrive `f008_actual.parquet` con:

- riga `MYCASH`: `date=oggi, qty=1, price=initial_amount, value=initial_amount`;
- per **ogni symbol** del portfolio: `qty=0, price=0, value=0` alla data odierna.

Serve a "partire" con solo cassa e tutte le posizioni a zero. In alternativa esiste
`init_portfolio.py` che compra notionalmente `initial_amount / n_symbols` per ogni
simbolo direttamente su Alpaca.

## 8. Report — `create_report()`

1. Legge `f006_position_new_qty.parquet`, droppa `qty_old, qty_diff, price`.
2. `cash = value_new` della riga `MYCASH`.
3. Ricarica i **prezzi correnti** con `get_latest_price` e ricalcola
   `value_new = qty_new * price` (valorizzazione a prezzi freschi).
4. Ripristina `value_new` della cassa.
5. `util.create_report(df)` genera il report HTML (vedi `02`).

## 9. Parametri di trading (dal conf.ini del portfolio)

| Parametro | Sezione | Default | Uso |
|---|---|---|---|
| `n_long` | `[portfolio]` | 5 | Numero di asset long nei pesi |
| `n_short` | `[portfolio]` | 5 | Numero di asset short nei pesi |
| `risk_percentage` | `[portfolio]` | 0.8 | Quota di equity investita |
| `qty_diff_perc_min` | `[portfolio]` | 0.2 | Soglia minima di variazione per tradare |
| `clip_min`/`clip_max` | `[predictions]` (modello) | -5/+5 | Range delle predizioni |

## 10. Osservazioni per la riscrittura

- La logica di transizione (tabella in §4) è la parte più delicata: va riscritta in
  modo testabile e con copertura unit test per tutte le combinazioni di segno.
- `execution()` e `define_orders()` duplicano la stessa matrice di transizioni:
  conviene un unico motore di "rebilanciamento" riutilizzato da entrambe.
- Il flusso mescola **posizioni cartacee** (yfinance, su parquet) e **posizioni
  reali** (Alpaca): il nuovo sistema deve scegliere una fonte di verità unica.
- La soglia `qty_diff_perc_min` evita churn (rimescolamento) quando le variazioni
  sono trascurabili.
- La logica `long → short` / `short → long` richiede due ordini atomici (close + open):
  in un nuovo sistema va gestito il rischio di esecuzione parziale.
