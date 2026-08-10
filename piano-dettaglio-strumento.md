# Piano Operativo: Dettaglio Strumento con Candlestick Chart

Questo documento definisce il piano per aggiungere una pagina di dettaglio strumento
raggiungibile cliccando una riga della Watchlist (`/watchlist/:symbol`), con candlestick
chart (lightweight-charts), KPI quantitativi, header strumento e notizie recenti.
Nessuna modifica allo schema MariaDB.

---

## Task 1: Backend — endpoint aggregato `GET /api/instruments/{symbol}`
- **Stato**: done
- **Scopo**: esporre in una sola chiamata meta strumento + ultimo segnale + storico OHLCV + notizie recenti.
- **Risultato atteso**: nuovo router `api/routers/instruments.py` (registrato in `api/main.py`) che
  restituisce `instrument`, `latest_signal` (con `vol_20` da `model_prediction`), `prices`
  (param `?days=`, default 250) e `news` (limit 10).
- **Test**: smoke test Python con `get_instrument_detail('AAPL')` e `('^TNX')`.
- **Todolist**:
  - [x] Creare `api/routers/instruments.py` con helper per latest signal e storico prezzi
  - [x] Registrare il router in `api/main.py`
  - [x] Gestire 404 per simbolo sconosciuto; `null` per segnale assente

## Task 2: Frontend — installare lightweight-charts + API service
- **Stato**: done
- **Scopo**: dipendenza chart e metodo API.
- **Risultato atteso**: `lightweight-charts` installato; `getInstrumentDetail(symbol, days?)` in `api.service.ts`.
- **Test**: `npm.cmd install lightweight-charts` completato.
- **Todolist**:
  - [x] `npm.cmd install lightweight-charts`
  - [x] Aggiungere `getInstrumentDetail(symbol, days?)` in `web/src/app/core/services/api.service.ts`

## Task 3: Frontend — pagina dettaglio `/watchlist/:symbol`
- **Stato**: done
- **Scopo**: vista dettaglio con header, KPI, candlestick chart e notizie recenti.
- **Risultato atteso**: header (simbolo, nome, tipo, area, settore, prezzo + variazione %),
  KPI box (Quant Score, Sentiment Mod, Segnale Finale, Vol 20), candlestick con volume +
  MA20/MA50 + selettore 3M/6M/1Y, notizie con badge impact cliccabili. Pattern signal,
  `ngOnDestroy` con `chart.remove()`.
- **Test**: `npm.cmd run build`.
- **Todolist**:
  - [x] Creare `web/src/app/features/instrument/instrument-detail.component.ts` (standalone, stile HUD)
  - [x] Registrare rotta `/watchlist/:symbol` in `app.routes.ts`
  - [x] Chart candlestick + volume + MA20/MA50 + periodi (ricampionamento 3M/6M/1Y)
  - [x] KPI box + header + notizie recenti
  - [x] Gestire `N/D` per segnali mancanti e stato `loading`

## Task 4: Frontend — righe watchlist cliccabili
- **Stato**: done
- **Scopo**: navigare al dettaglio dalla tabella Watchlist.
- **Risultato atteso**: righe cliccabili (`[routerLink]` + hover + cursor pointer) verso `/watchlist/:symbol`.
- **Test**: `npm.cmd run build`.
- **Todolist**:
  - [x] Aggiungere navigazione e stile hover alle righe in `watchlist.component.ts`

## Task 5: Documentazione
- **Stato**: done
- **Scopo**: aggiornare `doc-new-app/06-api-and-webapp.md`.
- **Risultato atteso**: endpoint `/api/instruments/{symbol}` in §1; vista "Dettaglio Strumento" in §2.2.
- **Test**: nessuno.
- **Todolist**:
  - [x] Riga endpoint in §1
  - [x] Vista dettaglio strumento in §2.2

