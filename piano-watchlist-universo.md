# Piano Operativo: Gestione Watchlist da Universo

Questo documento definisce il piano per rendere la pagina Watchlist gestibile:
aggiunta/rimozione di ticker, con le aggiunte scelte da un elenco universo
(pool di candidati non linkati al portafoglio).

**Decisioni confermate dall'utente**:
- Universo = strumenti `instrument` non linkati + seed dei ~100 simboli di
  training (UNIVERSE_SYMBOLS) non linkati al portafoglio.
- Rimozione = unlink (DELETE da `portfolio_instrument`): lo strumento resta
  nell'universo e può essere riaggiunto.
- Se un elemento della watchlist ha una posizione aperta, il bottone "Rimuovi"
  nella UI è **disabilitato** (e il backend rifiuta con 422 come fallback).

---

## Task 1: Seed universo candidati (batch job `universe`)
- **Stato**: done
- **Scopo**: popolare `instrument` con i ~100 simboli di training (da
  `train_universe_pipeline.py` `UNIVERSE_SYMBOLS`) **senza** linkarli al
  portafoglio, così il picker della UI ha candidati reali da cui scegliere.
- **Risultato atteso**: nuovo job CLI `python cli.py universe` idempotente che
  inserisce/aggiorna i simboli mancanti in `instrument` (active=1, metadata
  best-effort via yfinance) e NON tocca `portfolio_instrument`. Modifica di
  `seed_universe()` per non linkare più al portafoglio (il training legge già da
  `instrument WHERE active=1`).
- **Test**: eseguire `cli.py universe`; verificare con query che `instrument`
  contiene i simboli dell'universo e che `portfolio_instrument` resta invariato
  (solo watchlist attuale).
- **Todolist**:
  - [x] Creare `app/jobs/universe.py` con `run_universe_job()` che riusa
        `UNIVERSE_SYMBOLS` e la logica di fetch metadata
  - [x] Registrare il job `"universe"` in `app/cli.py` (`JOBS_MAP`)
  - [x] Modificare `seed_universe()` in `train_universe_pipeline.py` per non
        inserire più in `portfolio_instrument`

## Task 2: Endpoint API backend (universo + add/remove watchlist)
- **Stato**: done
- **Scopo**: esporre i candidati universo e consentire link/unlink da/to
  watchlist; esporre `has_open_position` per disabilitare "Rimuovi" nell'UI.
- **Risultato atteso**: in `api/routers/holdings.py`:
  - `GET /api/portfolios/{code}/universe` → strumenti attivi non linkati
    (symbol, name, type, area, sector, current_price), ordinati per area/symbol
  - `POST /api/portfolios/{code}/watchlist` body `{"instrument_id": N}` → insert
    in `portfolio_instrument` (idempotente), ritorna l'item aggiunto
  - `DELETE /api/portfolios/{code}/watchlist/{instrument_id}` → 422 se posizione
    aperta (`portfolio_position` con `qty != 0`); altrimenti unlink
  - Campo `has_open_position` in `_watchlist_rows`/`_serialize_watchlist_row`
- **Test**: smoke test via uvicorn/curl: lista universo, aggiunta, doppia
  aggiunta (no-op), rimozione con/senza posizione aperta.
- **Todolist**:
  - [x] Aggiungere `GET /universe` riusando il pattern di `_watchlist_rows`
  - [x] Aggiungere `POST /watchlist` con validazione strumento esistente/attivo
  - [x] Aggiungere `DELETE /watchlist/{instrument_id}` con guardia posizione aperta
  - [x] Aggiungere `has_open_position` a `_serialize_watchlist_row`

## Task 3: Frontend Angular — pagina Watchlist gestibile
- **Stato**: done
- **Scopo**: aggiungere/togliere ticker dalla pagina Watchlist; bottone
  "Rimuovi" disabilitato per righe con posizione aperta.
- **Risultato atteso**: in `watchlist.component.ts` (pattern signal): barra
  "Aggiungi dall'Universo" con `<select>` dei candidati + pulsante; pulsante
  "Rimuovi" per riga disabilitato se `has_open_position`; banner di stato per
  errori; reload dopo ogni operazione. `ApiService`: `getUniverse()`,
  `addToWatchlist()`, `removeFromWatchlist()`.
- **Test**: build dev (`npm run build`); flussi aggiunta/rimozione/blocco.
- **Todolist**:
  - [x] Aggiungere metodi in `api.service.ts`
  - [x] Caricare universo e gestire picker di aggiunta
  - [x] Colonna azione "Rimuovi" con `[disabled]="w.has_open_position"` e tooltip
  - [x] Verifica build: `npm run build`

## Task 4: Aggiornamento documentazione
- **Stato**: done
- **Scopo**: documentare il nuovo flusso universo→watchlist.
- **Risultato atteso**: aggiornati `doc-new-app/06-api-and-webapp.md` (nuovi
  endpoint + sezione UI watchlist) e `doc-new-app/08-portfolio-lifecycle.md` §4.
- **Test**: coerenza con gli altri doc.
- **Todolist**:
  - [x] Aggiornare tabella endpoint e sezione Watchlist in `06-api-and-webapp.md`
  - [x] Aggiornare §4 in `08-portfolio-lifecycle.md` (e job `universe` in `07-operativita-batch.md`)

## Task 5: Aggiunta nuovo simbolo all'universo (niente pagina extra)
- **Stato**: done
- **Scopo**: permettere di ampliare l'universo digitando un ticker nuovo
  direttamente dalla pagina Watchlist, senza una pagina dedicata.
- **Risultato atteso**: endpoint `POST /api/portfolios/{code}/universe`
  (`{"symbol": ...}`) che inserisce/riattiva lo strumento in `instrument`
  (metadata best-effort da yfinance, mai linkato), 409 se già in watchlist,
  422 se vuoto. Nella pagina Watchlist un campo "NUOVO SIMBOLO" con pulsante
  "+ Aggiungi all'universo": dopo l'aggiunta l'elemento viene auto-selezionato
  nel picker candidati. API e UI testate; build Angular OK.
- **Todolist**:
  - [x] Endpoint `POST /universe` in `api/routers/holdings.py`
  - [x] Metodo `addToUniverse` in `api.service.ts`
  - [x] Campo "NUOVO SIMBOLO" + handler in `watchlist.component.ts`
  - [x] Verifiche: add COST, re-add, 409 watchlist, 422 vuoto, cleanup
  - [x] `npm run build` + aggiornamento `06-api-and-webapp.md` / `08-portfolio-lifecycle.md`
