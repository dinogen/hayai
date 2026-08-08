# 08 — Operatività (CLI, cron, logging, notifiche)

Questo documento descrive come l'applicazione viene **eseguita** e **operata**:
interfaccia CLI, script di lancio, pianificazione cron, logging e notifiche.

## 1. Interfaccia CLI — `hayai.py`

```
python hayai.py -p <portfolio_id> [-i] [-s] [-n] [-r] [-e] [--init <amount>]
```

### Flag

| Flag | Lungo | Tipo | Effetto |
|---|---|---|---|
| `-p` | `--portfolio-id` | str (richiesto) | ID del portfolio da usare |
| `-i` | `--ingestion` | flag | Scarica quote + calcola features |
| `-s` | `--build-signals` | flag | Applica modello + definisce pesi |
| `-n` | `--new-position` | flag | Calcola nuova posizione/quantità/ordini |
| `-r` | `--report` | flag | Genera report |
| `-e` | `--execute-trades` | flag | Esegue i trade |
| | `--init` | float (default 0) | Inizializza portfolio con capitale |

### Note di implementazione

- Il codice chiama `args.parse_args()` **per ogni flag** (pattern difettoso ma
  funzionante): da correggere nella riscrittura (una sola chiamata).
- Le fasi sono **indipendenti**: lanciando `-s` senza `-i` si usano i parquet
  preesistenti. Lo stato deve essere persistito e coerente tra le fasi.
- Ogni esecuzione aggiorna il log `hayai.<data>.log` e (per `-s`, `-r`, `-e`)
  invia notifiche Telegram.

## 2. Script di lancio

### 2.1 `run.cmd` (Windows, locale)

Esegue in sequenza per `medium_tech_usa` e `mix_2`:
ingestion → build signals → new position → execute trades.

```
venv/Scripts/python hayai.py -p medium_tech_usa -i
venv/Scripts/python hayai.py -p medium_tech_usa -s
venv/Scripts/python hayai.py -p medium_tech_usa -n
venv/Scripts/python hayai.py -p medium_tech_usa -e
venv/Scripts/python hayai.py -p mix_2 -i
...
```

> Nota: `mix_2` non esiste nella cartella `data/` attuale (stato del repo): il
> comando fallirebbe in `create_context` (`FileNotFoundError`).

### 2.2 `run_ingestion.sh` (Linux/venv)

```
# lunedì 00:00
cd /opt/hayai
venv/bin/python hayai.py -p medium_tech_usa -i
venv/bin/python hayai.py -p eu -i
venv/bin/python hayai.py -p asia -i
venv/bin/python hayai.py -p medium_tech_usa -s
venv/bin/python hayai.py -p eu -s
venv/bin/python hayai.py -p asia -s
```

Esegue ingestion **e** build signals per i tre portfolio.

### 2.3 `run_trading.sh` (Linux/venv)

```
# lunedì 15:40
cd /opt/hayai
venv/bin/python hayai.py -p medium_tech_usa -n
venv/bin/python hayai.py -p eu -n
venv/bin/python hayai.py -p asia -n
```

Calcola la nuova posizione (senza eseguire i trade).

### 2.4 `run_report.sh` (Linux/venv)

```
# ogni giorno 06:00
venv/bin/python hayai.py -p medium_tech_usa -r
venv/bin/python hayai.py -p eu -r
venv/bin/python hayai.py -p asia -r
```

### 2.5 `cleanup.sh`

```
find . -name "*.log" -type f -mtime +7 -delete
```

Elimina i file di log più vecchi di 7 giorni.

## 3. Pianificazione (cron)

| Script | Frequenza | Orario | Fasi |
|---|---|---|---|
| `run_ingestion.sh` | settimanale (lunedì) | 00:00 | `-i` + `-s` |
| `run_trading.sh` | settimanale (lunedì) | 15:40 | `-n` |
| `run_report.sh` | giornaliero | 06:00 | `-r` |
| `cleanup.sh` | (non specificato) | — | pulizia log |

La sequenza settimanale è quindi: **lunedì 00:00** dati+segnali → **lunedì 15:40**
nuova posizione → **ogni giorno 06:00** report. L'esecuzione dei trade (`-e`) non è
pianificata negli script `sh` (presente solo in `run.cmd` locale).

> Da chiarire con la riprogettazione: chi, quando e con quale autorizzazione esegue
> `-e` in produzione.

## 4. Logging — `hayai_log.py`

- `create_logger(name)`:
  - livello `DEBUG`;
  - file handler: `hayai.<data>.log` (nella working directory);
  - console handler: `DEBUG`;
  - formato: `%(asctime)s - %(name)s - %(levelname)s - %(message)s`.
- Ogni modulo crea il proprio logger con `hayai_log.create_logger(__name__)`.
- `log_filename()` → `hayai.<date>.log`.

### Contenuto tipico dei log

- Avvio del processo e parametri (`hayai.py`).
- Progresso ingestion: `Processing {symbol} ({i}/{count})...`, salti per cache 4h.
- Applicazione modello: numero di asset e data.
- Definizione pesi/quantità/ordini: righe per asset con `qty_old`, `qty_new`,
  `qty_diff`.
- Esecuzione ordini: buy/sell/short per simbolo.
- Errori Alpaca (`APIError`) loggati ma non fatali.

> Nota: i log non sono strutturati (JSON), non hanno rotazione e sono scritti nella
> cwd. Il `cleanup.sh` rimuove i file più vecchi di 7 giorni. Il file di log dopo
> `-s` viene inviato via Telegram.

## 5. Notifiche Telegram

Inviate da `hayai_msg.py` (dettagli in `07`):

| Fase | Evento |
|---|---|
| `-s` | Invio del file di log |
| `-r` | Invio del report HTML (se generato) |
| `-e` | Messaggio testuale di conferma esecuzione |

## 6. Notebook operativi

- `control.ipynb`: compra notionalmente `100000 / n_symbols` per ogni simbolo del
  portfolio `medium_tech_usa` su Alpaca (paper) — usato come portafoglio "di
  controllo".
- `prova.ipynb`: esplorazioni (client Alpaca, yfinance forex, accesso file parquet,
  prezzi ask/bid). Non è parte del flusso di produzione.
- `sample_portfolio.ipynb`: campionamento del portfolio modello (vedi `02`).
- `init_portfolio.py`: script standalone che acquista una quota notional per ogni
  simbolo del portfolio (valori hardcoded `mix_2`, 100000).

## 7. Requisiti operativi per la riscrittura

- **Orchestrazione**: le fasi sono invocazioni CLI separate e dipendono da file
  intermedi su disco; un nuovo sistema batch dovrebbe gestire pipeline con stato,
  retry e idempotenza.
- **Scheduling**: esiste già un pattern settimanale (ingestion/signals lunedì,
  position lunedì pomeriggio, report giornaliero). Da mantenere/esplicitare.
- **Esecuzione ordini**: va separata da una logica di autorizzazione/conferma.
- **Logging**: da rendere strutturato, con rotazione e aggregazione.
- **Segreti**: oggi in file ini in chiaro nel repo (`.gitignore` a parte): da
  migrare a secret manager / variabili d'ambiente.
- **Notifiche**: il canale Telegram va disaccoppiato (webhook/event bus).
