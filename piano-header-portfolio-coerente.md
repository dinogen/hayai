# Piano Operativo: Header Coerente con le Tabelle (Portafoglio & Recommendations) (`piano-header-portfolio-coerente.md`)

Questo documento definisce il piano per rendere i **riquadri HUD** degli header delle pagine
"Portafoglio Attuale" e "Composizione Consigliata" **coerenti con le tabelle sottostanti**,
con tutti i valori calcolati **client-side in TypeScript** (reattivi alle modifiche dell'editor).

Decisioni prese con l'utente:
- **Riquadri LONG e SHORT** della pagina Portafoglio: mostrano **esposizione + P&L per lato**,
  derivati dalle righe dell'editor (`rows()`), quindi sempre sincronizzati con la tabella.
- **NAV** della pagina Portafoglio: **anteprima client-side** = `cash salvato + longValue − shortValue`
  (stessa matematica del backend `nav = cash + Σ market_value`), con badge "ANTEPRIMA" e delta quando
  la tabella è modificata ma non salvata. CASH resta invariato (cambia solo al salvataggio).
- **Pagina Recommendations**: stessa logica applicata al target, con due nuovi riquadri
  **TARGET LONG / TARGET SHORT** calcolati in TS dagli `items()` (composizione consigliata).

Nessuna modifica al backend: i calcoli riusano i dati già esposti dagli endpoint `/holdings` e
`/recommendations/latest`.

---

## Task 1: Metodi di Calcolo in TS — Pagina Portafoglio
- **Stato**: done
- **Scopo**: Aggiungere in `holdings.component.ts` i metodi per P&L per lato e NAV di anteprima,
  tutti derivati da `rows()` (reattivi alle modifiche dell'editor).
- **Risultato atteso**: esistono `longPnl()`, `shortPnl()`, `navPreview()`, `navDeltaVsSaved()`
  e `navDirty()`; `navPreview()` replica `cash + Σ market_value` del backend.
- **Test**: build Angular senza errori; a runtime i valori coincidono con il NAV salvato a pagina carica.
- **Todolist**:
  - [x] Aggiungere `longPnl()` = Σ `pnl(row)` per righe long.
  - [x] Aggiungere `shortPnl()` = Σ `pnl(row)` per righe short.
  - [x] Aggiungere `navPreview()` = `cash_balance` salvato + `longValue() − shortValue()`.
  - [x] Aggiungere `navDeltaVsSaved()` e `navDirty()` (soglia di tolleranza ~0.005€).

## Task 2: Template Header — Pagina Portafoglio
- **Stato**: done
- **Scopo**: Aggiornare i riquadri dell'header affinché riportino i valori della tabella.
- **Risultato atteso**:
  - Riquadri LONG e SHORT: valore esposto + sub-linea P&L del lato colorata (verde/rosso).
  - Riquadro NAV: mostra `navPreview()`; quando `navDirty()` mostra badge "ANTEPRIMA" e riga
    "salvato €X (Δ)" colorata. CASH invariato.
- **Test**: modifica di qty/prezzo in tabella → LONG/SHORT/P&L/NAV si aggiornano live; SALVA → badge sparisce.
- **Todolist**:
  - [x] Aggiornare riquadro LONG con `formatPnl(longPnl())`.
  - [x] Aggiornare riquadro SHORT con `formatPnl(shortPnl())`.
  - [x] Aggiornare riquadro NAV con anteprima + badge e delta.

## Task 3: Computed in TS — Pagina Recommendations
- **Stato**: done
- **Scopo**: Aggiungere in `recommendations.component.ts` i `computed()` per il target per lato,
  derivati da `items()` (le card della composizione consigliata).
- **Risultato atteso**: esistono `longTarget()`, `shortTarget()`, `longCount()`, `shortCount()`;
  `totalRecommended()` riusa la somma dei due lati.
- **Test**: build Angular senza errori; la somma long+short = `totalRecommended()`.
- **Todolist**:
  - [x] Aggiungere `longTarget()` / `shortTarget()` (Σ `target_amount` per side).
  - [x] Aggiungere `longCount()` / `shortCount()`.
  - [x] Refactor `totalRecommended()` per riusare i due lati.

## Task 4: Template Header — Pagina Recommendations
- **Stato**: done
- **Scopo**: Sostituire il riquadro "TARGET RACCOMANDATO" con i due riquadri per lato.
- **Risultato atteso**: header con riquadri **TARGET LONG** (verde) e **TARGET SHORT** (rosso),
  ciascuno con sub-linea numero di posizioni; `SCOSTAMENTO (NAV−TARGET)` resta sul totale.
- **Test**: verifica visiva su dati reali; somma dei due riquadri = target totale precedente.
- **Todolist**:
  - [x] Aggiungere riquadro TARGET LONG con conteggio posizioni.
  - [x] Aggiungere riquadro TARGET SHORT con conteggio posizioni.

## Task 5: Build e Verifica
- **Stato**: done
- **Scopo**: Verificare che il frontend compili e che i calcoli siano corretti.
- **Risultato atteso**: `npm run build` (o `ng build`) in `hayai-new/web` senza errori.
- **Test**: build; smoke test manuale delle due pagine.
- **Todolist**:
  - [x] Eseguire la build Angular.
  - [x] Correggere eventuali errori di compilazione.

## Task 6: Aggiornamento Documentazione
- **Stato**: done
- **Scopo**: Allineare i documenti di progetto alle nuove viste.
- **Risultato atteso**:
  - `doc-new-app/06-api-and-webapp.md`: §2.2/§2.3 descrivono i riquadri LONG/SHORT (Portafoglio) e
    TARGET LONG/SHORT (Recommendations) calcolati in TS e l'anteprima NAV client-side.
  - `doc-new-app/10-simulated-portfolio-value.md`: §5 aggiornata sui riquadri dell'header.
- **Todolist**:
  - [x] Aggiornare `doc-new-app/06-api-and-webapp.md`.
  - [x] Aggiornare `doc-new-app/10-simulated-portfolio-value.md`.
