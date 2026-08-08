# 09 — Vincoli e requisiti per la riscrittura

Questo documento raccoglie tutto ciò che serve per **riscrivere** l'applicazione in
un nuovo progetto con **processo batch + webapp**. È il documento di sintesi dei
vincoli di dominio, dei problemi del codice attuale e dei requisiti da preservare.

## 1. Vincoli di dominio (logica da preservare)

### 1.1 Pipeline dati → segnali → posizioni → esecuzione

La catena logica è:
1. **Ingestion**: storico OHLCV per simbolo → features + target.
2. **Segnali**: applicazione del modello sull'ultima data → predizione per asset.
3. **Pesi**: `prediction.clip / vol_20`, selezione top `n_long` long e bottom
   `n_short` short, normalizzazione a somma |pesi| = 1.
4. **Posizione**: `qty_new = round(weight_new * capital / price)`,
   `capital = equity * risk_percentage`, soglia minima `qty_diff_perc_min`.
5. **Ordini**: matrice di transizione di segno (`BUY`/`SELL`/`CLOSE`, con close+open
   per le inversioni).
6. **Esecuzione**: ordini su broker (oggi Alpaca paper) o simulazione locale.

### 1.2 Definizione del target

- `target = clip(ln(close_{t+trd} / close_t) / vol_20, clip_min, clip_max)`
- Il modello predice il **rendimento futuro normalizzato per volatilità**, non un
  prezzo.
- Look-ahead: il target "guarda avanti" di `target_return_days`; va calcolato senza
  guardare il futuro quando si valida in backtest.

### 1.3 Model portfolio vs trading portfolio

- Il **modello** viene addestrato su un universo ampio (3-4 paesi, 3-4 settori,
  ~1000 simboli). I **portfolio di trading** sono sottoinsiemi.
- Le dummies country/sector usano le **categorie del modello** (CategoricalDtype):
  in questo modo le colonne restano allineate tra training e inference anche se nel
  portfolio di trading manca una categoria.
- Ogni portfolio dichiara il modello da usare (`[predictions] model`) oppure è
  esso stesso un modello (id che inizia con `model_`).

### 1.4 Duale sorgente dati

- `data_source` = `yfinance` o `alpaca`, configurabile per portfolio.
- Forex e indici sono **sempre** da yfinance (configurati nel modello).
- Alpaca non fornisce `high/low` nello storico → incompatibile con `hl_range` e
  `close_range` (vedi §2).

## 2. Problemi e bug rilevati nel codice attuale

### 2.1 Bug funzionali

| # | Problema | Dettaglio |
|---|---|---|
| B1 | **Flusso `-n` rotto con `data_source=alpaca`** | `get_actual_position_alpaca` restituisce solo i simboli detenuti e **senza** colonna `value_old` né riga `MYCASH`. `build_new_position` perde gli asset da aprire (merge `left` su `df_old`) e `define_new_quantity` va in `KeyError: 'value_old'` (`row_mycash['value_old'].iloc[0]`). Il flusso funziona solo con yfinance. |
| B2 | **`get_equity_yfinance` non funzionante** (`hayai_dao.py:154`) | Legge `actual_positions.parquet` (nome hardcoded) e accede a `qty_old` che non esiste in quel file (le colonne sono `symbol, qty, price, value`). Inoltre somma **quantità** (`qty_old`) come fossero valori monetari. `get_equity` non è usato nel flusso principale ma è un trap. |
| B3 | **Vendite/short non simulabili in modalità yfinance** | `_place_order_buy_yfinance` simula solo gli acquisti. `place_order_sell` e `place_order_short` usano comunque il `TradingClient` Alpaca. La modalità yfinance è quindi incoerente per il rebilanciamento. |
| B4 | **`hl_range`/`close_range` NaN con Alpaca** | Alpaca non fornisce `high/low`; `add_financial_features` produce NaN per queste due feature. Nel training il `dropna` elimina le righe, in inference le NaN restano e degradano la predizione. |
| B5 | **Nomi file hardcoded** | `execution()`, `_place_order_buy_yfinance`, `get_equity_yfinance` usano stringhe (`position_new_qty.parquet`, `actual_positions.parquet`) invece delle costanti `FILE_*`. Il file `actual_positions.parquet` non coincide con `f008_actual.parquet`. |
| B6 | **`argparse.parse_args()` chiamato più volte** | In `hayai.py` per ogni flag; funziona ma è un anti-pattern da correggere. |
| B7 | **Argomenti scambiati in `place_order_buy`** | Firma `place_order_buy(symbol, qty, tc)` ma chiamata `place_order_buy(client, symbol, qty_new)` in `hayai_bo.py:454` → con `alpaca` il primo buy crasha (TypeError su `assert qty > 0` con qty stringa). `place_order_sell`/`place_order_short` sono corretti. |
| B8 | **Segno incoerente di `qty` negli ordini** | `BUY`/`SELL` usano `abs(qty)`; `CLOSE` usa `qty_old` grezzo → per la chiusura di uno short `qty` è negativa. Chi consuma `f007_orders.parquet` deve gestirlo. |

### 2.2 Questioni di correttezza/robustezza

| # | Problema | Dettaglio |
|---|---|---|
| R1 | **Divisione per zero sui prezzi** | In `define_new_quantity` il merge `outer` con `fillna(0)` può dare `price=0` per simboli senza prezzo → `qty_new = round(value_new / 0)` → `inf`. Non gestito. |
| R2 | **Allineamento colonne features** | La normalizzazione in inference assume che le colonne di f001 coincidano con `mins.csv`/`maxs.csv` del modello. Se cambiano i set forex/indexes o le categorie, la predizione degenera (NaN) senza errori espliciti. |
| R3 | **`label_min`/`label_max` obbligatori senza fallback** | Manca un valore di default: mancanza → crash di lettura. |
| R4 | **Errore nelle asserzioni** | `define_weight` usa `assert` per la somma dei pesi: con `python -O` gli assert vengono disabilitati (controllo saltato). Meglio una validazione esplicita. |
| R5 | **Ordini parziali non gestiti** | Nelle inversioni (close + open) un errore API su Alpaca viene solo loggato e si prosegue: rischio di posizioni "scoperte". |
| R6 | **Convenzione nomi parquet duplicata** | Esistono in cartella sia file `f0NN_*` che file legacy (`features.parquet`, `positions.parquet`, ...): fonte di confusione per chi rilegge i dati. |
| R7 | **Logging non strutturato** | File di testo con date nel nome, nessuna rotazione, niente JSON/correlation id. |
| R8 | **Nessun test** | Il repository non contiene test automatici: la logica di trading (matrice di transizione) non ha copertura. |

### 2.3 Issues dal Readme.md

1. ✔ Separare i conf.ini; chiavi nei file secret.ini — **fatto**.
2. ✔ Togliere la colonna `volume` dalle feature — **fatto**.
3. Per scaricare i dati servono solo `symbol, close, volume` — **in parte**:
   le feature `hl_range`/`close_range` richiedono `high/low`.
4. ✔ Creare più modelli, uno per portfolio, ognuno indica il modello che usa —
   **fatto** (convenzione `model_*` + `[predictions] model`).
5. Funzione che calcola posizione ed equity dai dati scaricati — **non completata**
   (`get_equity_yfinance` rotta, vedi B2).
6. ✔ Report con posizione calcolata in loco — **fatto**.
7. ✔ Download dati da yfinance o alpaca in base a parametro (anche Europei) —
   **fatto** ma con limitazioni (vedi B4).

## 3. Requisiti per la nuova architettura (batch + webapp)

### 3.1 Componenti batch (job)

1. **Job ingestion** (per portfolio):
   - aggiorna `hist/{symbol}` (con cache/configurazione del refresh window);
   - ricalcola features + target;
   - valida completezza (min righe, assenza di NaN nelle feature critiche).
2. **Job training** (per modello):
   - ingresso: features del model portfolio;
   - output: artefatto del modello **versionato** (architettura, pesi, mins/maxs,
     label_min/max, metriche, dataset fingerprint).
3. **Job signals** (per portfolio):
   - applica il modello all'ultima data → predizioni;
   - produce pesi target.
4. **Job rebalance** (per portfolio):
   - calcola nuova posizione (equity, capital, qty_new, qty_diff, soglia minima);
   - genera ordini (matrice di transizione);
   - **dry-run** (preview) prima dell'esecuzione reale.
5. **Job execution**:
   - esegue ordini sul broker (con conferma/authorization);
   - registra stato ordini e riepilogo.
6. **Job report**:
   - valorizza la posizione a prezzi correnti;
   - genera report (HTML/PDF) e notifiche.

### 3.2 Webapp

- **Dashboard portfolio**: posizione corrente, equity, pesi target vs attuali,
  ordini in attesa/eseguiti, storico.
- **Dashboard modelli**: versioni, metriche, data di training.
- **Amministrazione**: configurazione portfolio (conf), abilitazione job,
  approvazione ordini (gate manuale prima dell'esecuzione).
- **Audit**: log di ogni step con input/output file, tempi, errori.

### 3.3 Requisiti funzionali estratti

- **Idempotenza**: rieseguire un job deve dare lo stesso risultato (nessuna
  doppia contabilizzazione di ordini).
- **Riproducibilità**: dati, parametri e modello versionati; la normalizzazione
  min-max deve far parte dell'artefatto del modello.
- **Fonte di verità unica** per la posizione (oggi c'è ambiguità parquet vs Alpaca).
- **Separazione** tra "calcolo della posizione" (deterministico, off-line) e
  "esecuzione" (broker, con autorizzazione).
- **Gestione errori** per API broker: retry, dead-letter, stato ordine.
- **Configurazione tipizzata** e validata (oggi è un dict piatto non tipizzato).
- **Test automatici** sulla matrice di transizione degli ordini (tutte le
  combinazioni di segno di `qty_old`/`qty_new`).

### 3.4 Requisiti non funzionali

- **Performance**: evitare `yf.Ticker.info` per-simbolo; usare batch o un provider
  con API bulk (oggi ~1000 simboli → centinaia di chiamate).
- **Segreti**: migrare da file ini in chiaro (anche se gitignorati) a secret manager.
- **Osservabilità**: log strutturati con rotation, metriche per job, notifiche su
  errore.
- **Dati**: decidere il provider unico o la strategia per OHLCV completo (high/low).

## 4. Raccomandazioni architetturali

- **Batch**: separare chiaramente i job (ingestion/training/signals/rebalance/
  execution/report) come unità indipendenti eseguibili e testabili, orchestrate da
  un runner (es. scheduler, DAG) con stato persistente.
- **Modello**: artefatto unico versionato (id, versione, pesi, normalizzazione,
  metadati) referenziato dai portfolio — niente dipendenza da cartelle di file.
- **Dati**: repository unico con schema esplicito (un DB o un lake versionato)
  invece di parquet sparsi con doppie convenzioni di nome.
- **Domain**: la logica di trading (pesi, posizioni, ordini) deve essere pura e
  priva di side-effect, così da poter essere condivisa tra job batch e API della
  webapp.
- **Esecuzione**: "rebalance plan" calcolato deterministicamente → approvato →
  eseguito. Il plan è un artefatto rilevabile (JSON), non solo ordini su file.

## 5. Checklist di validazione per la riscrittura

- [ ] Il calcolo dei pesi riproduce esattamente `define_weight` (top n_long/bottom
      n_short, somma |pesi|=1).
- [ ] La matrice di transizione ordini copre tutte le 10 combinazioni di §4 doc 05
      con test unitari.
- [ ] L'equity viene calcolata in modo corretto e coerente (no somme di quantità).
- [ ] Le feature `hl_range`/`close_range` hanno una strategia definita quando
      `high/low` non sono disponibili.
- [ ] La normalizzazione dell'inference è allineata al training (stesse colonne,
      stessi min/max, stesso ordine).
- [ ] Il flusso supporta sia `yfinance` che `alpaca` con comportamento identico e
      documentato per le differenze note.
