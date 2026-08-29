# Piano Operativo: Esecuzione Raccomandazioni per Riga (rimozione bottone, flip long/short)

Questo documento definisce il piano per modificare il flusso di esecuzione delle raccomandazioni:
1. rimuovere il bottone "Applica Raccomandazioni del Modello" dalla pagina Portafoglio;
2. aggiungere un bottone "Esegui" per ogni riga della tabella di riconciliazione nella pagina Raccomandazioni;
3. garantire che il passaggio long→short (o viceversa) chiuda la posizione e ne apra una nuova (vendi/compra).

**Scoperta chiave dell'analisi**: `build_trades()` in `app/portfolio_rebalance.py` gestisce GIÀ il flip
long↔short correttamente (chiusura con `sell`/`cover` + riapertura con `buy`/`short`, righe 74-97). Il
problema reale è che la **tabella di riconciliazione** in `api/routers/portfolios.py` (righe 149-190)
confronta le quantità in **valore assoluto ignorando la side** (`owned_side`/`target_side` non vengono usati),
producendo messaggi errati e incoerenti con l'esecuzione reale. Il piano rende la riconciliazione derivata
dalla stessa logica di esecuzione (sorgente unica di verità).

Decisioni concordate:
- L'endpoint "Esegui" rilegge l'ultima raccomandazione dal DB (server authoritative).
- Blocco se raccomandazione stale (default 4 giorni, come il job `align`).
- Esecuzione della singola riga = target esatto (soglia non applicata; le righe `hold` hanno bottone disabilitato).
- Nella pagina Portafoglio resta il riquadro "Data segnale" (si rimuove solo il bottone).

---

## Task 1: Rimuovere il bottone "Applica Raccomandazioni del Modello" dalla pagina Portafoglio
- **Stato**: done
- **Scopo**: eliminare il bottone e tutta la logica correlata dal componente holdings, mantenendo il flusso manuale (editor + SALVA) e il riquadro "Data segnale".
- **Risultato atteso**: nessun riferimento residuo a `applyRecommendations`/`hasRecommendations` nel componente; la pagina resta funzionante per la gestione manuale.
- **Test**: `npm run build` in `hayai-new/web` senza errori; navigazione manuale su `/portfolio` senza errori console.
- **Todolist**:
  - [x] Rimuovere il blocco HTML del bottone in `holdings.component.ts` (righe 73-77).
  - [x] Rimuovere il metodo `applyRecommendations()` (righe 416-446).
  - [x] Rimuovere il metodo `hasRecommendations()` (righe 309-311).
  - [x] Rimuovere il signal `recommendations` (riga 194) e la sua assegnazione in `loadData()` (riga 223); mantenere `recDate`.
  - [x] Aggiornare il testo dello stato vuoto (riga 146) rimuovendo il riferimento al bottone.
  - [x] Verificare la build Angular.

## Task 2: Rendere la riconciliazione coerente con l'esecuzione (flip long/short)
- **Stato**: done
- **Scopo**: rifattorizzare il calcolo della riconciliazione perché usi la stessa logica di `build_trades()` (quantità con segno), producendo azioni `buy/sell/short/cover/flip/hold` e messaggi identici a ciò che l'esecuzione farà davvero.
- **Risultato atteso**: per un flip long→short la tabella mostra "chiudi long e apri short N" (2 trade: `sell` + `short`); per un flip short→long "chiudi short e apri long N" (`cover` + `buy`). Nessun messaggio con quantità assolute fuorvianti.
- **Test**: nuovi unit test pytest (funzione pura, senza DB) per i casi: apertura long/short, chiusura long/short, incremento, riduzione, flip, hold sotto soglia, arrotondamento short.
- **Todolist**:
  - [x] Aggiungere in `app/portfolio_rebalance.py` una helper `build_reconciliation(current, target, close_map, threshold_eur=None)` che deriva azione/messaggio per strumento dai trade prodotti da `build_trades()` (nessuna duplicazione di logica).
  - [x] Nella helper, arrotondare i target short con `round_short_qty` (come già fa `align_portfolio_to_recommendations`), saltando i target che arrotondano a 0.
  - [x] Definire i messaggi: `apri long/short N`, `compra N`, `shorta N`, `vendi N`, `copri N`, `chiudi long/short (vendi/copri N)`, `chiudi long e apri short N` / `chiudi short e apri long N`, `mantieni (invariato)`.
  - [x] Sostituire il blocco di calcolo riconciliazione in `api/routers/portfolios.py` (righe 149-190) con la nuova helper, mantenendo lo stesso output API (instrument_id, symbol, name, instrument_type, owned_qty, owned_side, target_qty, target_side, current_price, action, message, diff).
  - [x] Installare pytest nel venv (`venv\Scripts\python -m pip install pytest`) e scrivere unit test in `hayai-new/tests/` per `build_reconciliation`.
  - [x] Verificare l'endpoint `GET /api/portfolios/{code}/recommendations/latest` su casi long/short/flip.

## Task 3: Nuovo endpoint POST `/portfolios/{code}/holdings/execute`
- **Stato**: done
- **Scopo**: eseguire l'allineamento di un singolo strumento alla sua ultima raccomandazione, rileggendo il target dal DB, con guardia anti-stale e riuso di `build_trades()` + `apply_trades()`.
- **Risultato atteso**: la POST applica i trade (incluso il flip come 2 operazioni), aggiorna `portfolio_trade`/`portfolio_position`/`portfolio_cash` atomicamente e ritorna il riepilogo. Rifiuta con 422/409 se: strumento non in watchlist, nessuna raccomandazione, target 0, short che arrotonda a 0, o raccomandazione stale.
- **Test**: chiamate manuali all'API sui casi (flip, apertura, chiusura, già allineato, stale) e verifica in DB di trade/posizione/cash.
- **Todolist**:
  - [x] Aggiungere il modello Pydantic `ExecuteRecommendationRequest { instrument_id: int }` in `api/routers/holdings.py`.
  - [x] Implementare `execute_recommendation()`: validazioni (watchlist, raccomandazione esistente, target>0, short arrotondato), guardia stale (`stale_days` default `DEFAULT_STALE_DAYS`, parametro query opzionale), lookup posizione corrente e ultimo prezzo, `build_trades(threshold_eur=None)`, `apply_trades()` in transazione.
  - [x] Gestire il caso "già allineato" (nessun trade) con risposta informativa (200, `executed: false`).
  - [x] Registrare l'endpoint nel router holdings (già montato in `api/main.py`).
  - [x] Test manuale dell'API e verifica su DB (EMLC: trade `buy` registrato, snapshot posizione e cash aggiornati; stato ripristinato con rollback).

## Task 4: Frontend — bottone "Esegui" nella tabella di riconciliazione
- **Stato**: done
- **Scopo**: aggiungere una colonna con bottone "Esegui" per ogni riga della riconciliazione nella pagina Raccomandazioni, con stato di caricamento per-riga, reload dopo l'esecuzione e messaggi di esito.
- **Risultato atteso**: cliccando "Esegui" si chiama il nuovo endpoint; dopo il successo la tabella (e i valori NAV) si aggiornano; bottone disabilitato su righe `hold` e durante l'esecuzione.
- **Test**: `npm run build` senza errori; navigazione manuale su `/recommendations` ed esecuzione di una riga (verifica aggiornamento tabella + DB).
- **Todolist**:
  - [x] Aggiungere in `core/services/api.service.ts` il metodo `executeRecommendation(code, instrumentId)` → POST `/portfolios/{code}/holdings/execute`.
  - [x] Estrarre il caricamento dati di `ngOnInit()` in un metodo `loadData()` riutilizzabile.
  - [x] Aggiungere nella tabella riconciliazione (righe 136-155) una colonna con bottone "Esegui", disabilitato se `row.action === 'hold'` o mentre una riga è in esecuzione.
  - [x] Aggiungere `executingId` e `status` signal + handler `executeRow(row)` con conferma, chiamata API, reload e messaggi di successo/errore.
  - [x] Estendere i colori dei badge per le nuove azioni `short`, `cover`, `flip` (mantenendo `buy` verde, `sell` rosso, `hold` grigio).
  - [x] Verificare la build Angular.

## Task 5: Aggiornamento della documentazione
- **Stato**: done
- **Scopo**: allineare `doc-new-app/` al nuovo flusso (bottone rimosso, esecuzione per riga, nuovo endpoint).
- **Risultato atteso**: nessun riferimento residuo al bottone rimosso; nuovo endpoint e nuova semantica di riconciliazione documentati.
- **Test**: grep su `doc-new-app/` per confermare che non restino riferimenti a "Applica Raccomandazioni del Modello" come pulsante della pagina Portafoglio.
- **Todolist**:
  - [x] `06-api-and-webapp.md`: rimuovere il riferimento al pulsante "Applica Raccomandazioni del Modello" (riga 144), documentare `POST /holdings/execute` e la riconciliazione con gestione flip.
  - [x] `08-portfolio-lifecycle.md`: aggiornare il flusso manuale (riga 63) indicando l'esecuzione per riga dalla pagina Raccomandazioni.
  - [x] `10-simulated-portfolio-value.md`: aggiornare i riferimenti (righe 8, 46, 195).
  - [x] `11-maintenance-manual.md`: aggiornare la tabella dei flussi (riga 176).
