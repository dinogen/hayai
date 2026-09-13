# 11 — Manuale Operativo

Riferimento rapido per chi mantiene HAYAI v2 in sviluppo su Windows e in produzione su Raspberry Pi.
Per le specifiche complete consulta i documenti dedicati in `doc-new-app/`.

---

## 1. File Batch Windows (`.bat`)

Script di avvio manuale per l'ambiente di sviluppo/test su Windows.
In produzione (Raspberry Pi) l'esecuzione è affidata a cron (vedi `07-operativita-batch.md`).
Tutti i file sono nella root del progetto (tranne `train_v3.bat`, dentro `hayai-new/`).

| File | Cosa fa | Quando lanciarlo |
|---|---|---|
| `avvia_mariadb_server.bat` | Avvia il server MariaDB in console (`mariadbd.exe`) | **Sempre per primo**: senza DB nulla funziona. Lasciare la finestra aperta |
| `avvia_backend.bat` | Avvia FastAPI (uvicorn, `--reload`) su `127.0.0.1:8000` | Prima di usare la webapp in dev |
| `avvia_frontend.bat` | Avvia il dev server Angular su `http://localhost:4200` | In sviluppo (richiede backend avviato) |
| `avvia_ciclo_completo.bat` | Pipeline notturna completa: `data → news → sentiment → predict → signal → recommend → summaries` | Per simulare il ciclo notturno a mano o dopo un fermo |
| `scarica_dati.bat` | Identico al ciclo completo con `--portfolio main` esplicito e `pause` finale | Alternativa ad `avvia_ciclo_completo.bat` per il ripristino dopo downtime |
| `avvia_prediction.bat` | Solo la coda di segnale: `predict → signal → recommend` | Quando prezzi/notizie sono già aggiornati e si vuole ricalcolare solo la composizione |
| `train_universe.bat` | Ingestion dati (100 asset, 5 anni) + training modello MLP | Ogni 2-3 mesi per riaddestrare il modello |
| `hayai-new\train_v3.bat` | Training completo di una versione sperimentale (`v3`) | Per sperimentare una nuova versione prima di renderla attiva |

**Ordine tipico in dev**: `avvia_mariadb_server.bat` → `avvia_ciclo_completo.bat` → `avvia_backend.bat` → `avvia_frontend.bat`

> I batch attivano il venv (`venv\Scripts\activate`) e impostano `PYTHONPATH=hayai-new`.

---

## 2. Checklist di Primo Soccorso

| Sintomo | Cosa controllare |
|---|---|
| Nessun dato nuovo | `job_run` per lo stato dei job; log in `logs/cron.log` e `logs/hayai.log` |
| Job `data` fallito | Connessione internet/yfinance (rate limit); i dati mancanti vengono recuperati automaticamente al run successivo (upsert idempotente). Il client yfinance (`app/yf_client.py`) ritenta con backoff esponenziale su HTTP 429/5xx; se il blocco persiste, il job logga l'errore e il ciclo continua |
| Job `metadata` con `429 Too Many Requests` | Rate limit Yahoo su `quoteSummary`; i metadati restano quelli esistenti (`metadata_date` invariato) e vengono ritentati al prossimo run |
| Nessuna raccomandazione | Verificare che `predict`/`signal` siano andati a buon fine; `model_registry` deve avere un modello `active` |
| Sentiment assente | Controllare `NEWS_LLM_ENABLED` in `.env` e le credenziali DeepSeek |
| NAV fermo | Il job `nav` gira solo se il portafoglio ha posizioni; cash/posizioni si muovono solo con operazioni manuali in webapp |
| Webapp vuota in dev | Avviare in ordine: `avvia_mariadb_server.bat` → `avvia_backend.bat` → `avvia_frontend.bat` |
