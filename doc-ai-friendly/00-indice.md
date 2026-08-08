# HAYAI — Documentazione AI-friendly

## Scopo

Questo progetto è una vecchia applicazione Python monolitica che:

1. Scarica dati finanziari da **Yahoo Finance** e **Alpaca**;
2. Addestra un modello predittivo (Keras/Deep Learning);
3. Genera segnali e pesi per un portafoglio;
4. Calcola posizioni e ordini di trading;
5. Esegue i trade su Alpaca (o li simula con yfinance);
6. Invia notifiche e report via **Telegram**.

L'obiettivo di questa documentazione è estrarre **tutte le logiche** dal codice sorgente
in modo esauriente (livello code-level), così da poter riscrivere l'applicazione in un
nuovo progetto con architettura moderna: **processo batch + webapp**.

La documentazione è scritta in **italiano**. Il codice di riferimento è in inglese.

## Sezioni

| Sezione | Documenti | Stato |
|---|---|---|
| **Analisi del sistema attuale** (legacy HAYAI) | `01`–`10` | completata |
| **Progettazione nuova app** (batch + webapp su Raspberry) | `11`–`17` | completata |

La nuova applicazione: database **MariaDB**, job batch con **cron**, backend
**FastAPI**, frontend **Angular**, dati e notizie da **yfinance**, modello Keras
addestrato su PC e inferito su Raspberry (ONNX). **Nessun broker/ordini reali**:
il sistema produce solo **raccomandazioni** long/short.

## Come leggere i documenti

| Documento | Contenuto |
|---|---|
| `01-architettura-attuale.md` | Panoramica dei moduli, del flusso CLI e della pipeline end-to-end |
| `02-pipeline-dati.md` | Ingestione dati e schema di ogni file parquet (f001–f008, hist) |
| `03-feature-engineering.md` | Tutte le formule delle feature e definizione del target |
| `04-modello-predittivo.md` | Training del modello (normalizzazione, architettura, metriche) e inference |
| `05-logica-trading.md` | Pesi, posizioni, quantità, ordini, esecuzione trade |
| `06-configurazione.md` | Schemi di conf.ini, secret.ini, portfolio.csv e struttura cartelle |
| `07-integrazioni-esterne.md` | API esterne: Yahoo Finance, Alpaca, Telegram |
| `08-operativita-cron.md` | CLI, script batch, cron, logging, notifiche |
| `09-vincoli-riscrittura.md` | Problemi noti e requisiti per la riscrittura (batch + webapp) |
| `10-glossario.md` | Glossario dei termini chiave |
| `11-architettura-target.md` | **Nuova app**: componenti (MariaDB, batch, FastAPI, Angular), flusso, decisioni, deploy Pi |
| `12-requisiti-funzionali.md` | **Nuova app**: requisiti funzionali/non funzionali e user stories |
| `13-schema-database.md` | **Nuova app**: schema MariaDB (tabelle, colonne, indici, upsert) |
| `14-pipeline-ml.md` | **Nuova app**: training su PC, artefatto ONNX, inferenza, raccomandazioni |
| `15-pipeline-notizie.md` | **Nuova app**: notizie yfinance e riassunti markdown |
| `16-api-e-webapp.md` | **Nuova app**: API FastAPI e viste Angular |
| `17-operativita-batch.md` | **Nuova app**: job batch, cron, deploy passo-passo su Raspberry |

## Mappa codice → documenti (sezione "nuova app")

| Area | Documenti |
|---|---|
| Architettura e decisioni | `11` |
| Requisiti | `12` |
| Schema DB | `13` |
| Modello ML | `14` |
| Notizie | `15` |
| API + frontend | `16` |
| Batch + deploy | `17` |

## Mappa codice → documenti

| File sorgente | Documento principale |
|---|---|
| `hayai.py` | `01`, `08` |
| `hayai_util.py` | `01`, `06` |
| `hayai_dao.py` | `02`, `07` |
| `hayai_bo.py` | `03`, `05` |
| `hayai_trade.py` | `05`, `07` |
| `hayai_msg.py` | `07`, `08` |
| `hayai_log.py` | `08` |
| `train_model.ipynb` | `04` |
| `sample_portfolio.ipynb` | `02`, `06` |
| `control.ipynb`, `prova.ipynb` | `07` (spunti esplorativi) |
| `init_portfolio.py` | `05` |
| `run.cmd`, `*.sh`, `cleanup.sh` | `08` |
| `Readme.md` | `02`, `09` |

## Stato della documentazione

- Tutti i documenti sono derivati dall'analisi del codice sorgente nel repository.
- Ogni formula, parametro e transizione di trading è stata verificata contro il sorgente.
- I nomi dei file in codice sono riportati esattamente come nel sorgente (inclusi gli
  hardcoded come `position_new_qty.parquet`).
- I documenti `11`–`17` descrivono la **progettazione target** della nuova app
  (batch + webapp su Raspberry), derivata dall'analisi del legacy ma non vincolata
  al suo codice.
