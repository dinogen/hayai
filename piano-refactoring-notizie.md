# Piano Operativo: Refactoring Sentiment Notizie — Sorpresa, Durata, Decay, Propagazione

Questo documento definisce il piano per evolvere l'analisi delle notizie da un
sentiment a 3 bucket (`bullish/neutral/bearish`) a un modello basato su
`impact_score` continuo (-5..+5), durata dell'effetto, confidenza-gate,
decadimento temporale e propagazione delle notizie macro per area.
Metodo di riferimento dell'utente: `appunti-notizie.md`.

---

## Task 1: Schema Database (news_sentiment + portfolio_signal)

- **Stato**: todo
- **Scopo**: Migrare `news_sentiment` all'impact_score continuo con durata e
  superficie di impatto; aggiungere `sentiment_breakdown` JSON a `portfolio_signal`
  per esporre alla webapp il dettaglio per-notizia che ha contribuito al segnale.
- **Risultato atteso**: `sql/schema.sql` aggiornato (fresh install) + file
  `sql/migration_news_sentiment_refactor.sql` (ALTER su DB esistente) +
  `doc-new-app/02-database-schema.md` allineato.
- **Todolist**:
  - [ ] Aggiornare tabella `news_sentiment` in `schema.sql`: drop `sentiment`,
        aggiungere `impact_score DECIMAL(3,1)`, `impact_duration ENUM('brief','medium','long')`,
        `impact_surface VARCHAR(255)`.
  - [ ] Aggiungere colonna `sentiment_breakdown JSON NULL` a `portfolio_signal`.
  - [ ] Creare `sql/migration_news_sentiment_refactor.sql` con ALTER/CREATE per DB esistenti.
  - [ ] Aggiornare `doc-new-app/02-database-schema.md` (sezioni 2.10 e 2.11).

## Task 2: Job sentiment (nuovo prompt "sorpresa vs attese")

- **Stato**: todo
- **Scopo**: Riscrivere il prompt DeepSeek in `app/jobs/sentiment.py` per estrarre
  `impact_score`, `impact_duration`, `impact_surface`, `confidence`, `catalyst`,
  `rationale_it`, istruendo il modello a confrontare esito vs attese del mercato e a
  ridurre la confidenza quando il confronto non è esplicito nel testo.
- **Risultato atteso**: `sentiment.py` scrive in `news_sentiment` con i nuovi campi;
  parsing robusto con clamping a ±5.0 e validazione della durata.
- **Todolist**:
  - [ ] Riscrivere il prompt (checklist sorpresa/attese, catena causale, chi guadagna/perde).
  - [ ] Parsing e validazione di `impact_score` (clamp -5..5), `impact_duration`, `impact_surface`.
  - [ ] Aggiornare `INSERT INTO news_sentiment` con i nuovi campi.

## Task 3: Job signal (gate, decay, propagazione, breakdown)

- **Stato**: todo
- **Scopo**: Riscrivere l'aggregazione in `app/jobs/signal.py`: confidenza come gate
  (soglia 0.30), contributo = `(impact_score/5) × 0.20 × confidence × decay`,
  decay per durata (brief=24h, medium=96h, long=336h), propagazione notizie macro
  tramite `impact_surface` alle aree, e salvataggio del breakdown JSON.
- **Risultato atteso**: `signal.py` produce `final_signal = quant_score + modificatore`
  con clamping ±0.20 e popola `sentiment_breakdown`.
- **Todolist**:
  - [ ] Funzione `impact_decay(duration, age_hours)`.
  - [ ] Query notizie con propagazione per `impact_surface` (join su `instrument.area`).
  - [ ] Calcolo contributi con gate e decay, clamp ±0.20.
  - [ ] Popolare `sentiment_breakdown` JSON e aggiornare l'upsert.

## Task 4: Job summaries (formattazione markdown)

- **Stato**: todo
- **Scopo**: Adattare `app/jobs/summaries.py` ai nuovi campi (impact_score, durata).
- **Risultato atteso**: Markdown con emoji per segno/magnitudo e indicazione della durata.
- **Todolist**:
  - [ ] Aggiornare query SELECT e formattazione per `impact_score` e `impact_duration`.

## Task 5: API REST (news + signals con breakdown)

- **Stato**: todo
- **Scopo**: Aggiornare `api/routers/portfolios.py`: gli endpoint news restituiscono
  `impact_score`, `impact_duration`, `impact_surface`; l'endpoint signals restituisce
  anche `sentiment_breakdown` (dettaglio per-notizia per la webapp).
- **Risultato atteso**: risposte API coerenti con il nuovo schema.
- **Todolist**:
  - [ ] Endpoint `/portfolios/{code}/news` e `/news/{id}`: nuovi campi.
  - [ ] Endpoint `/portfolios/{code}/signals`: includere `sentiment_breakdown`.

## Task 6: Webapp Angular (news, news-detail, signals)

- **Stato**: todo
- **Scopo**: Adattare le viste al punteggio continuo: badge colore da `impact_score`,
  filtro "solo con analisi" su `impact_score`, dettaglio per-notizia nella pagina
  signals (tooltip/dettaglio con impact_score, durata, decay).
- **Risultato atteso**: build webapp OK; la pagina signals mostra il dettaglio
  per-notizia che ha contribuito al modificatore.
- **Todolist**:
  - [ ] `news.component.ts`: colori/filtri basati su `impact_score` + durata.
  - [ ] `news-detail.component.ts`: pannello analisi con score, durata, superficie, confidenza.
  - [ ] `signals.component.ts`: colonne separate + dettaglio espandibile per notizia (score, durata, decay).

## Task 7: Documentazione (04 e 06)

- **Stato**: todo
- **Scopo**: Allineare la documentazione del progetto.
- **Risultato atteso**: `04-news-llm-pipeline.md` con nuovo prompt e formula di
  aggregazione/decay; `06-api-and-webapp.md` con i nuovi campi; riferimento a
  `appunti-notizie.md`.
- **Todolist**:
  - [ ] Aggiornare `doc-new-app/04-news-llm-pipeline.md`.
  - [ ] Aggiornare `doc-new-app/06-api-and-webapp.md`.

## Task 8: Verifica finale

- **Stato**: todo
- **Scopo**: Smoke test dei job e build webapp.
- **Risultato atteso**: nessun errore di import/sintassi; `npm run build` OK.
- **Todolist**:
  - [ ] Compilare i moduli Python (`python -m compileall`).
  - [ ] Unit smoke su decay/gate (funzione pura).
  - [ ] `npm run build` della webapp.
