# Piano Operativo: Job Settimanale di Allineamento Portafoglio (`piano-job-settimanale-align.md`)

Questo documento definisce il piano per introdurre un **nuovo job batch settimanale** che allinea
automaticamente il portafoglio simulato alle raccomandazioni del modello. Il job **non fa parte del
ciclo giornaliero**: viene schedulato una volta a settimana, il **martedì alle 15:00** (riga cron
`20 15 * * 2`, dopo il refresh prezzi intraday delle 15:00 per usare prezzi freschi).

Caratteristiche decise con l'utente:
- **Soglia di tolleranza**: rispetta `rebalance_threshold_eur` (default €50) — le variazioni
  same-direction sotto soglia restano invariate (hold), evitando micro-operazioni.
- **Guardia anti-stale**: se l'ultima `rec_date` è più vecchia di 4 giorni (weekend/ferie) il job si
  ferma senza operare (stato `skipped`), a meno di `--force`.
- **Solo allineamento**: nessun refresh dati interno; si usano i prezzi e le raccomandazioni già in DB.
- **Modulo condiviso**: la logica di generazione trade oggi vive in `save_holdings`
  (`api/routers/holdings.py`); viene estratta in un modulo riusabile da job e API (no duplicazione).

---

## Task 1: Modulo Condiviso di Allineamento (`app/portfolio_rebalance.py`)
- **Stato**: done
- **Scopo**: Creare un modulo Python riusabile con la logica di generazione trade e applicazione
  allineamento (portafoglio → raccomandazioni), con supporto della soglia di tolleranza.
- **Risultato atteso**: `app/portfolio_rebalance.py` con:
  - `build_trades(current, target, close_map, threshold_eur=None) -> (trades, desired)` che
    riproduce la logica attuale di `save_holdings` (chiusura full, open/incremento, sign-flip =
    close+open, short interi via `round_short_qty`) e applica la soglia solo sugli aggiustamenti
    same-direction (open/close sempre eseguiti).
  - `apply_trades(...)` che in transazione scrive `portfolio_trade`, snapshot `portfolio_position`
    e upsert `portfolio_cash` (`initial_capital + Σ amount`).
  - `align_portfolio_to_recommendations(portfolio_code, stale_days=4, force=False) -> dict` che
    legge l'ultima `rec_date` (`MAX(rec_date)`), applica la guardia anti-stale, costruisce il target
    dai `target_qty` (skip null/0 e fuori watchlist) e ritorna il riepilogo (rec_date, trades, cash,
    NAV, threshold, skip).
- **Test**: `python -m app.cli align --portfolio main` (via Task 2/5); verifica dei trade/posizioni/cash
  in DB; confronto NAV.
- **Todolist**:
  - [x] Creare `app/portfolio_rebalance.py` con `build_trades`, `apply_trades`, `align_portfolio_to_recommendations`.
  - [x] Gestire short interi, soglia `threshold_eur`, guardia anti-stale e skip strumenti non idonei.
  - [x] Correggere la direzione dei trade di riduzione same-direction (long→sell, short→cover; il
        vecchio `save_holdings` emetteva buy/short anche riducendo).

## Task 2: Job CLI `align` (`app/jobs/align.py` + `app/cli.py`)
- **Stato**: done
- **Scopo**: Esporre l'allineamento come job batch CLI registrato in `JOBS_MAP` con nome `align`,
  schedulabile da cron.
- **Risultato atteso**: `python -m app.cli align [--portfolio main] [--days 4] [--force]` esegue
  l'allineamento, logga esito e dettagli in `job_run`.
- **Todolist**:
  - [x] Creare `app/jobs/align.py` con `run_align_job(portfolio_code="main", days=4, force=False)`.
  - [x] Registrare `"align"` in `JOBS_MAP` in `app/cli.py`.
  - [x] Aggiungere branch dispatch in `cli.py` che passa `days` e `force` al job `align`.

## Task 3: Refactor `save_holdings` per Riusare il Modulo
- **Stato**: done
- **Scopo**: Eliminare la duplicazione facendo riusare a `POST /portfolios/{code}/holdings/save`
  (`api/routers/holdings.py`) le funzioni condivise `build_trades`/`apply_trades`, mantenendo il
  comportamento attuale (allineamento alla lettera, senza soglia).
- **Risultato atteso**: `save_holdings` mantiene le validazioni (side, qty>0, watchlist, duplicati)
  e delega la generazione trade + scrittura al modulo condiviso con `threshold_eur=None`. Risposta
  API invariata (con in più la correzione delle riduzioni same-direction).
- **Test**: POST manuale all'endpoint o riuso dei dati esistenti; build/import senza errori.
- **Todolist**:
  - [x] Sostituire la logica interna di `save_holdings` con le chiamate al modulo condiviso.
  - [x] Verificare che i dettagli di risposta (`nav`, `cash_balance`, `positions_value`, `trades_executed`) restino coerenti.

## Task 4: Cron e Aggiornamento Documentazione
- **Stato**: done
- **Scopo**: Documentare il nuovo job e la riga cron settimanale, aggiornando i documenti di progetto.
- **Risultato atteso**:
  - `doc-new-app/07-operativita-batch.md`: job `align` nell'elenco (con sezione non-daily) + riga cron
    `20 15 * * 2` (commento: dopo il refresh intraday delle 15:00).
  - `doc-new-app/10-simulated-portfolio-value.md`: §2.1 aggiornata (allineamento anche automatico settimanale).
  - `doc-new-app/08-portfolio-lifecycle.md`: menzione dell'allineamento settimanale del martedì.
- **Todolist**:
  - [x] Aggiornare `07-operativita-batch.md` (job list + cron).
  - [x] Aggiornare `10-simulated-portfolio-value.md` §2.1.
  - [x] Aggiornare `08-portfolio-lifecycle.md`.

## Task 5: Verifica End-to-End su DB Locale
- **Stato**: done
- **Scopo**: Verificare che il job funzioni correttamente senza alterare i dati di modello.
- **Risultato atteso**: esecuzione `venv\Scripts\python -m app.cli align --portfolio main` su DB
  locale; `job_run` con stato `success`/`skipped`; trade, posizioni, cash e NAV coerenti; test
  `--force` sulla guardia anti-stale; nessun errore d'import in `cli.py`/`holdings.py`.
- **Todolist**:
  - [x] Eseguire il job in `hayai-new` e ispezionare `job_run`/tabelle (già allineato → `already_aligned`).
  - [x] Verificare NAV/cash/posizioni e il comportamento con soglia e anti-stale (`--days 0` → `stale`, `--force` → bypass).
  - [x] Test end-to-end con trade su portafoglio temporaneo (open/reduce/close long e short) + `save_holdings` via API; cleanup eseguito.
  - [x] Controllare che l'API importi senza errori (smoke test TestClient su holdings/value/recommendations).


