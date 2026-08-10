# Piano Operativo: Watchlist Arricchita (Area + Segnale Modello + Coefficiente News + Volatilità)

Questo documento definisce il piano per arricchire la watchlist di HAYAI v2 con una
nuova pagina dedicata che mostra, per ogni strumento: area geografica, ultimo segnale
del modello (`quant_score`), ultimo coefficiente dalle news (`llm_sentiment_modifier`),
segnale ibrido finale (`final_signal`), volatilità a 20 giorni (`vol_20`, coscienza del
rischio) e prezzo corrente. Strumenti senza segnale mostrano `N/D` grigio.

Nessuna modifica allo schema MariaDB: i dati esistono già in `instrument`,
`portfolio_signal` e `model_prediction`.

---

## Task 1: Backend — watchlist arricchita (helper + endpoint)
- **Stato**: done
- **Scopo**: esporre nell'API i campi `area`, `sector`, `quant_score`, `llm_sentiment_modifier`,
  `final_signal`, `signal_date`, `vol_20` e `current_price` per ogni strumento della watchlist.
- **Risultato atteso**: nuovo endpoint `GET /api/portfolios/{code}/watchlist`; la `watchlist`
  dell'endpoint `/holdings` espone gli stessi campi (il dropdown attuale ignora i campi extra).
  `vol_20` proviene da `model_prediction` (ultima `as_of_date` del modello attivo).
- **Test**: avvio uvicorn + `Invoke-RestMethod` su `/api/portfolios/main/watchlist` e `/holdings`.
- **Todolist**:
  - [x] In `api/routers/holdings.py`: helper `_active_model_id()` (stessa logica di `signal.py`)
  - [x] In `api/routers/holdings.py`: helper `_watchlist_rows(portfolio_id)` con LEFT JOIN a
        `portfolio_signal` (ultimo `signal_date`) e `model_prediction` (ultima `as_of_date`)
  - [x] In `api/routers/holdings.py`: serializzazione con valori `None` per strumenti senza segnale
  - [x] Usare l'helper in `get_holdings` (watchlist payload) e nel nuovo endpoint `/watchlist`
  - [x] Ordinamento di default per `area`, poi `symbol`

## Task 2: Frontend — nuova pagina Watchlist
- **Stato**: done
- **Scopo**: pagina Angular dedicata `/watchlist` con tabella HUD.
- **Risultato atteso**: colonne Strumento, Tipo, Area (badge colorato), Quant Score,
  Sentiment Mod, Segnale Finale, Vol 20 (rischio) e Prezzo; `N/D` grigio per valori mancanti.
  Rispetta il vincolo change-detection a `signal` (doc 06 §2.0).
- **Test**: `npm.cmd build` in `hayai-new/web`.
- **Todolist**:
  - [x] Aggiungere `getWatchlist(code)` in `web/src/app/core/services/api.service.ts`
  - [x] Creare `web/src/app/features/watchlist/watchlist.component.ts` (standalone, style Cyber Light HUD)
  - [x] Registrare rotta `/watchlist` in `web/src/app/app.routes.ts`
  - [x] Aggiungere link "Watchlist" nella navbar (`navbar.component.ts`)
  - [x] Formatting JetBrains Mono, colori `#16a34a`/`#dc2626`, badge area

## Task 3: Documentazione
- **Stato**: done
- **Scopo**: aggiornare `doc-new-app/06-api-and-webapp.md`.
- **Risultato atteso**: endpoint `/watchlist` nella tabella API §1; nuova vista Watchlist in §2.2.
- **Test**: nessuno (solo testo).
- **Todolist**:
  - [x] Aggiungere riga endpoint `/api/portfolios/{code}/watchlist` in §1
  - [x] Aggiungere la vista "Watchlist" a §2.2 con l'elenco colonne
