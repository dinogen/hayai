# Piano Operativo: Metadati Asset da Yahoo Finance — Settore (Comparto), Country e Area (EU/USA/Asia/Emergenti)

Questo documento definisce il piano per arricchire ogni strumento della watchlist
(`instrument`) con i metadati scaricati da Yahoo Finance tramite **yfinance**:
il **settore/comparto** (`sector`, fallback `category` per gli ETF) e la
**Country** (`country`), dalla quale si deriva l'**area** di appartenenza
(`EU`, `USA`, `Asia`, `Emergenti`, `Altro`).

L'obiettivo è disporre di un filtro geografico/settoriale affidabile sull'universo
investibile, utile per la lettura delle raccomandazioni e per eventuali analisi di
diversificazione, senza introdurre Forex o nuove fonti dati.

Decisioni di design confermate:
- Fonte: `yf.Ticker(symbol).info` (stessa session `requests` già usata dal job `data`).
- `sector`: per le azioni si usa `info['sector']`; per ETF/bond_yield si usa
  `info['category']` come fallback (l'ETF non ha un settore merceologico).
- `country`: da `info['country']` (stringa normalizzata, es. "United States").
- `area`: derivata dalla `country` con regola a **priorità** (Emergenti > EU > USA > Asia > Altro),
  perché Cina/India/Brasile sono geograficamente in Asia/Americhe ma nel contesto
  di portafoglio contano come Emergenti.
- Nuove colonne su `instrument`: `sector VARCHAR(128)`, `country VARCHAR(128)`,
  `area ENUM('usa','eu','asia','emerging','other')` e `metadata_date DATE`
  (data dell'ultimo fetch per evitare download ripetuti).
- Nuovo job batch `metadata` (CLI), schedulabile via cron dopo il job `data`.

---

## Task 1: Schema Database — Colonne Metadati su `instrument` e Aggiornamento Documentazione
- **Stato**: done
- **Scopo**: Estendere la tabella `instrument` con i metadati (settore, country, area, data fetch) e allineare documentazione e seed.
- **Risultato atteso**: `hayai-new/sql/schema.sql` e `doc-new-app/02-database-schema.md` contengono le nuove colonne; DDL `ALTER TABLE` eseguito sul MariaDB locale e verificato con `DESCRIBE instrument`.
- **Todolist**:
  - [x] Definire le nuove colonne: `sector VARCHAR(128) NULL`, `country VARCHAR(128) NULL`, `area ENUM('usa','eu','asia','emerging','other') NULL`, `metadata_date DATE NULL` (aggiunte alla fine della tabella per non rompere i seed esistenti).
  - [x] Aggiornare `hayai-new/sql/schema.sql` (CREATE TABLE `instrument` con le nuove colonne).
  - [x] Aggiornare `doc-new-app/02-database-schema.md` (DDL e sezione descrittiva con la regola di derivazione area e la priorità Emergenti > EU > USA > Asia > Altro).
  - [x] Eseguire l'`ALTER TABLE instrument ADD COLUMN ...` sul MariaDB locale e verificare con `DESCRIBE instrument` / `SHOW CREATE TABLE`.

## Task 2: Modulo Mappatura Area — `app/area.py`
- **Stato**: done
- **Scopo**: Centralizzare la derivazione dell'area a partire dalla country, con tabella di mappatura normalizzata e funzione riutilizzabile (`map_area`).
- **Risultato atteso**: `hayai-new/app/area.py` con `map_area(country: str) -> str` che ritorna uno tra `usa`, `eu`, `asia`, `emerging`, `other`, con gestione di valori mancanti/casuali (default `other`) e normalizzazione (lowercase, senza punteggiatura/accapo).
- **Todolist**:
  - [x] Creare il dizionario `AREA_BY_COUNTRY` con i paesi mappati: EU (Italia, Germania, Francia, Spagna, Paesi Bassi, Irlanda, Svizzera, Svezia, Danimarca, Belgio, Austria, Portogallo, Finlandia, UK...), USA (United States), Asia (Giappone, Corea del Sud, Taiwan, Hong Kong, Singapore, Australia, India...), Emergenti (Cina, Brasile, India, Sudafrica, Messico, Russia, Indonesia, Turchia...).
  - [x] Implementare `map_area(country)` con priorità **Emergenti > EU > USA > Asia > Altro** (un paese presente in più liste vince secondo l'ordine, es. Cina → `emerging`).
  - [x] Normalizzare l'input (strip, lowercase, rimozione di punteggiatura) prima della lookup.
  - [x] Scrivere una mini-suite di verifica (`python -m app.area` o un piccolo script di smoke test) con casi noti: `United States → usa`, `Italy → eu`, `Japan → asia`, `China → emerging`, `Brazil → emerging`, `xyz → other`.

## Task 3: Job Batch `metadata` — Download Settore/Country e Upsert su `instrument`
- **Stato**: done
- **Scopo**: Nuovo job che per ogni strumento attivo della watchlist scarica `ticker.info`, estrae `sector`/`category` e `country`, deriva l'area e aggiorna `instrument` (solo se mancante o obsoleto, oppure forzato).
- **Risultato atteso**: `hayai-new/app/jobs/metadata.py` con `run_metadata_job(portfolio_code)`; job registrato in `hayai-new/app/cli.py` come `metadata`; log in `job_run` e conteggi in output (fetch effettuati, aggiornati, falliti).
- **Todolist**:
  - [x] Leggere strumenti attivi come in `app/jobs/data.py` (JOIN `instrument` + `portfolio_instrument` + `portfolio`).
  - [x] Usare `requests.Session` con User-Agent e delay ~2s tra gli strumenti (pattern già esistente in `data.py`).
  - [x] Estrarre da `info` (con accesso sicuro `.get()`): `sector` (fallback `category` per ETF), `country`; gestire `info` vuoto o errore senza interrompere il job (log + conteggio falliti).
  - [x] Calcolare `area` con `app.area.map_area` (fallback per simbolo quando `country` manca); impostare `metadata_date = CURDATE()`.
  - [x] Upsert su `instrument` (`UPDATE ... SET sector, country, area, metadata_date, updated_at`) con query `UPDATE` e `ROW_COUNT()` per capire quali righe cambiano davvero.
  - [x] Salvare solo se la metadata manca/è più vecchia di N giorni (default 30) o con flag `--force` (aggiungere opzione in CLI se serve).
  - [x] Registrare `"metadata": run_metadata_job` in `JOBS_MAP` in `app/cli.py`.

## Task 4: Esecuzione Backfill e Verifica su MariaDB
- **Stato**: done
- **Scopo**: Popolare i metadati per l'universo iniziale (seed: 20 strumenti) e verificare che sector/country/area siano corretti.
- **Risultato atteso**: `SELECT symbol, sector, country, area, metadata_date FROM instrument` mostra tutti gli strumenti della watchlist valorizzati; eventuali valori mancanti (`other`/NULL) documentati e riconciliati.
- **Todolist**:
  - [x] Eseguire `venv\Scripts\python -m app.cli metadata --portfolio main` (prima volta senza `--force`).
  - [x] Controllare `job_run` per il job `metadata` (status success e dettagli).
  - [x] Verificare il backfill con `SELECT` e verificare i casi attesi (AAPL → usa, VGK → eu/other, EEM → emerging se presente, ^TNX → usa).
  - [x] Per gli strumenti rimasti `other`/NULL, verificare a mano l'`info` di yfinance e, se il campo manca davvero, decidere se aggiungere una mappatura manuale di fallback (es. per bond yield `^TNX` → usa).

## Task 5: Esposizione API e Documentazione Finale
- **Stato**: done
- **Scopo**: Rendere disponibili settore/country/area nelle API e documentare il flusso, così che il frontend (e il promotore) possano leggere l'area e il settore degli strumenti.
- **Risultato atteso**: `GET /api/portfolios/{code}` (o l'endpoint di lista strumenti esistente) restituisce `sector`, `country`, `area`; `doc-new-app/06-api-and-webapp.md` e `08-portfolio-lifecycle.md` aggiornati; nessuna regressione sulle altre API.
- **Todolist**:
  - [x] Aggiornare la query di lista strumenti in `api/routers/portfolios.py` (SELECT con `sector`, `country`, `area`).
  - [x] (Se utile) aggiungere filtro opzionale `?area=eu` all'endpoint strumenti.
  - [x] Aggiornare `doc-new-app/06-api-and-webapp.md` con i nuovi campi/filtro.
  - [x] Aggiornare `doc-new-app/08-portfolio-lifecycle.md` (o `01`) citando il job `metadata` nel ciclo notturno/weekly.
  - [x] Smoke test API: `GET /api/portfolios/main` ritorna i nuovi campi; `ng build` (frontend) senza errori se l'interfaccia non usa campi rimossi.
