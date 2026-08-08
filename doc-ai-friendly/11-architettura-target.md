# 11 — Architettura target (batch + webapp su Raspberry)

Questo documento è l'**hub** della progettazione della nuova applicazione. Descrive
componenti, flussi, decisioni di design e modalità di deploy su un **Raspberry Pi**
che funge da webservice. È il riferimento da cui partire per scrivere il nuovo
progetto. Il codice legacy (HAYAI) resta solo come **archivio** di riferimento
concettuale (documenti 01–10).

## 1. Obiettivo e scope

Nuova applicazione che:

1. **aggiorna ogni giorno** i dati di mercato (prezzi OHLCV, forex, indici) e le
   **notizie** relative ai portafogli via **yfinance**;
2. genera **riassunti in markdown** delle notizie per portafoglio;
3. esegue l'**inferenza** di un modello Keras (addestrato su PC) per ottenere
   **predizioni**;
4. calcola la **composizione consigliata** del portafoglio in **long/short**
   (solo raccomandazioni, nessun ordine reale su broker);
5. espone tutto via **webapp** (FastAPI + Angular) consultabile dal Raspberry.

Decisioni vincolanti (confermate dall'utente):

| Decisione | Scelta |
|---|---|
| Esecuzione ordini | **Solo raccomandazioni** (nessun broker) |
| Training modello | **Su PC** (Jupyter), artefatto deployato sul Pi |
| Fonte dati | **yfinance** (prezzi, forex, indici, notizie) — Alpaca eliminato |
| Obbligazioni | **ETF obbligazionari** (es. BND, TLT) + **rendimenti di Stato** (^TNX, ^FVX, ^TYX) |
| Tipi di asset | Azioni, ETF, valute (fx), rendimenti obbligazionari |
| Database | **MariaDB** (sul Raspberry) |
| Batch | **cron** sul Raspberry |
| Backend | **FastAPI** (uvicorn) |
| Frontend | **Angular** (statico, servito da nginx) |
| Hardware | Raspberry Pi 4/5, **installazione nativa** |

## 2. Architettura dei componenti

```
                    ┌──────────────────────────────────────────────┐
                    │              RASPBERRY PI                    │
                    │                                              │
  Internet ──► yfinance (prezzi, forex, indici, notizie)           │
                    │                                              │
                    │  ┌───────────┐   cron    ┌───────────────┐  │
                    │  │ BATCH PY  │◄──────────│  cronie       │  │
                    │  │ (job/...) │           └───────────────┘  │
                    │  └─────┬─────┘                              │
                    │        │ read/write                         │
                    │        ▼                                    │
                    │  ┌─────────────────────────┐                │
                    │  │      MariaDB            │                │
                    │  │ (prices, news, models,  │                │
                    │  │  predictions, reco)     │                │
                    │  └───────────┬─────────────┘                │
                    │              │                              │
                    │              ▼                              │
                    │  ┌────────────────────┐  ┌───────────────┐  │
                    │  │  FastAPI (uvicorn) │  │ nginx (static │  │
                    │  │  REST API /api     │  │ + reverse pr.)│  │
                    │  └─────────┬──────────┘  └───────┬───────┘  │
                    │            │                     │          │
                    │            └─────────┬───────────┘          │
                    │                      ▼                      │
                    │              Angular SPA (dist)             │
                    │  + artefatti modello + summary markdown     │
                    └──────────────────────────────────────────────┘

  PC (training):
  Jupyter → dataset multi-asset → Keras → export ONNX + normalizzazione → deploy Pi
```

Componenti:

1. **Database MariaDB** — unica fonte di verità dei dati (vedi `13-schema-database.md`).
2. **Processi batch (Python)** — schedulati via cron; ogni job è un comando CLI
   indipendente e idempotente (vedi `17-operativita-batch.md`).
3. **Backend FastAPI** — API REST in sola lettura per la webapp (vedi
   `16-api-e-webapp.md`).
4. **Frontend Angular** — SPA statica servita da nginx (build di produzione).
5. **Storage artefatti** — cartella sul Pi per `model.onnx`, `mins.csv`, `maxs.csv`,
   metadati, e riassunti markdown generati (duplicati anche in DB).
6. **Configurazione e segreti** — file `.env` (non versionato) per credenziali
   DB/Telegram; nessuna chiave broker.

## 3. Flusso end-to-end ("giorno-tipo")

### 3.1 Notte (cron, batch)

1. **data** — scarica prezzi OHLCV di tutti gli strumenti attivi (download batch
   yfinance), forex e indici; **upsert** in `price_daily`/`fx_rate`/`index_value`.
2. **news** — scarica le notizie per i simboli dei portafogli (`Ticker.news`,
   `yf.Search`), dedup e insert in `news`.
3. **summaries** — genera i riassunti markdown per portafoglio/giorno → `news_summary`.
4. **features** — calcola le feature per ogni strumento (adattate dal doc `03`,
   senza dummies country/sector per classi non-azionarie).
5. **predict** — normalizza con i parametri del modello e applica l'inferenza
   (onnxruntime) → `prediction`.
6. **recommend** — calcola i pesi target long/short → `recommendation` e posizioni
   target di massima (solo indicazione).

### 3.2 Giorno (webapp)

- L'utente accede alla webapp (LAN o esposta via VPN/tunnel).
- La SPA chiama FastAPI per: dashboard, prezzi, predizioni, composizione consigliata,
  notizie e riassunti.

## 4. Decisioni di design e giustificazioni

### 4.1 Perché yfinance come unica fonte

- Copre azioni, ETF, forex (`EURUSD=X`), indici e **rendimenti obbligazionari**
  (`^TNX`, `^FVX`, `^TYX`, `^IRX`).
- Espone le **notizie** (`Ticker.news`, `yf.Search`) senza chiave API.
- Gratuito, coerente con l'esistente, nessuna credenziale da gestire.

Limiti e mitigazioni (vedi `07` e `17`):
- **Rate limit**: usare `download` con più simboli in una chiamata, cache locale,
  retry con backoff, esecuzione in finestra notturna.
- **`Ticker.info` lento**: non usarlo per batch estesi; usare `download` per i
  prezzi e `get_news`/`Search` per le notizie.
- **ToS**: uso personale/non commerciale; verificare i termini Yahoo.

### 4.2 Perché ONNX per l'inferenza sul Raspberry

- Il modello è un piccolo MLP (Dense 100/80/20 → 1, ~41k params): adatto a CPU ARM.
- **Export Keras → ONNX** in fase di training (PC) e inferenza con **onnxruntime**
  (ruote `arm64` disponibili): niente TensorFlow sul Pi, footprint minimo.
- Fallback: Keras 3 con backend leggero (numpy) se l'export ONNX desse problemi.
- Il file `.keras` resta l'artefatto di training; l'ONNX è l'artefatto di runtime.

### 4.3 Perché MariaDB

- Supporto ARM64 ufficiale (`apt install mariadb-server`).
- Transazionale e affidabile per upsert idempotenti.
- SQL per query di serie storiche e aggregazioni (indici).
- Volumi piccoli (qualche centinaio di strumenti × 5 anni) → nessun tuning critico.

### 4.4 Perché FastAPI + Angular

- **FastAPI**: Python (stessa lingua dei batch), OpenAPI automatico, async,
  leggero per uvicorn su ARM.
- **Angular**: SPA standalone (Angular 17+), build statica servita da nginx,
  niente Node runtime sul Pi (solo build su PC o cross-compile).

### 4.5 Modelli per classe di asset (decisione chiave)

Le feature legacy usano dummies `country_*`/`sector_*` (doc `03`), inadatte a forex
e obbligazioni. La nuova pipeline usa **un feature set type-agnostic** comune
(log_return, momentum, volatilità, z-score, trend, volume quando disponibile,
forex/indici) e valuta **un modello per classe di asset** (stocks/etf, fx,
bond-yield) oppure un modello unico senza dummies. Dettagli in `14-pipeline-ml.md`.

### 4.6 Niente broker, niente ordini

- Il batch produce solo **raccomandazioni** (pesi target, posizioni desiderate).
- La webapp le presenta; nessun modulo di esecuzione né credenziali broker.
- Il `job_run` registra ogni esecuzione per audit.

## 5. Deploy su Raspberry Pi (panoramica; dettagli in `17`)

- OS: Raspberry Pi OS (Bookworm) **64-bit**.
- Installazione nativa: `mariadb-server`, `nginx`, Python venv (`/opt/hayai-new/venv`).
- Backend: servizio **systemd** `hayai-api` (uvicorn), bind su `127.0.0.1:8000`.
- Frontend: build Angular → `/var/www/hayai`, nginx reverse proxy `/api` → uvicorn.
- Batch: crontab di un utente dedicato (`hayai`) con i job di `17`.
- Backup: `mysqldump` giornaliero + copia artefatti.

## 6. Cosa viene "tradotto" dal legacy (mappa)

| Logica legacy | Nuova collocazione |
|---|---|
| Ingestione quote (doc `02`) | Batch `data` → `price_daily` |
| Feature engineering (doc `03`) | Batch `features` (type-agnostic) |
| Training Keras (doc `04`) | Notebook PC → artefatto ONNX |
| Inference (doc `04`) | Batch `predict` (onnxruntime) |
| Pesi long/short (doc `05`) | Batch `recommend` → `recommendation` |
| Posizioni/quantità target (doc `05`) | `recommendation` (solo indicazione) |
| Ordini/execution (doc `05`) | **eliminato** (niente broker) |
| Report HTML (doc `02`, `05`) | Viste webapp Angular |
| Notifiche Telegram (doc `08`) | Opzionale: alert job in `job_run` |
| Cron shell (doc `08`) | Cron Python (job CLI) |

## 7. Requisiti trasversali

- **Idempotenza**: ogni job è ripetibile senza effetti doppi (upsert per chiave
  naturale data+simbolo; `job_run` per stato).
- **Riproducibilità**: artefatti modello versionati (id+versione in
  `model_registry`); parametri espliciti in configurazione.
- **Osservabilità**: log strutturati per job + `job_run` in DB + alert opzionale.
- **Sicurezza**: nessuna chiave nel repo; `.env` locale; DB in ascolto su
  `127.0.0.1`; API in sola lettura.
- **Performance**: query indicizzate, batch notturni, frontend statico.
