# Piano Operativo: Rendere la pagina Notizie leggibile e gestibile

Questo documento definisce il piano per trasformare la pagina Notizie della webapp
in una vista compatta e filtrabile, aggiungere la pulizia automatica dei dati e
una pagina di dettaglio per ogni singola notizia.

---

## Task 1: Nuovo endpoint news con metadati e filtri
- **Stato**: todo
- **Scopo**: arricchire `GET /api/portfolios/{code}/news` includendo `sector`, `area`, `sentiment`, `confidence`, con parametri `?days=14&sector=&symbol=&limit=50`.
- **Risultato atteso**: API che restituisce notizie filtrabili, senza logica nel frontend.
- **Todolist**:
  - [ ] Modificare `api/routers/portfolios.py` aggiungendo join con `instrument` (sector, area) e parametri di filtro
  - [ ] Aggiungere parametro `days` (default 14) con clausola `published_at >= ...`
  - [ ] Ordinamento per `published_at DESC`, limit con paginazione

## Task 2: Nuovo job di pulizia `cleanup`
- **Stato**: todo
- **Scopo**: eliminare notizie/sentiment più vecchi di 14 giorni e i parquet cache scaduti in `tmp/`.
- **Risultato atteso**: job `python -m app.cli cleanup` (parametro `--days`, default 14) che cancella `news` (cascade su `news_sentiment`) e i file `*_news.parquet`/`*_gnews.parquet` più vecchi di 14gg.
- **Todolist**:
  - [ ] Creare `app/jobs/cleanup.py` con `run_cleanup_job`
  - [ ] Registrare il job in `app/cli.py` (JOBS_MAP)
  - [ ] Aggiungere riga cron in `doc-new-app/07-operativita-batch.md`

## Task 3: Frontend — nuova vista news a card per settore
- **Stato**: todo
- **Scopo**: sostituire la pagina monolitica con card compatte raggruppate per settore, titolo cliccabile, badge sentiment, filtri e "carica altri".
- **Risultato atteso**: pagina `/news` leggibile (titolo + publisher + data + badge) con gruppi per settore e pulsante "Mostra altre notizie".
- **Todolist**:
  - [ ] Riscrivere `web/src/app/features/news/news.component.ts` usando `getNews()` con filtri e raggruppamento per settore
  - [ ] Card con badge sentiment (🟢/🟡/🔴) e data relativa
  - [ ] Filtri (settore, simbolo, solo con sentiment) e paginazione incrementale

## Task 4: Frontend — pagina dettaglio notizia
- **Stato**: todo
- **Scopo**: rotta `/news/:id` con titolo, publisher, data, summary e link alla fonte originale.
- **Risultato atteso**: cliccando su una card si apre la pagina dettaglio (fetch `GET /news/{id}`).
- **Todolist**:
  - [ ] Nuovo endpoint `GET /news/{id}` in `api/routers/portfolios.py`
  - [ ] Nuovo componente `news-detail.component.ts` + rotta in `app.routes.ts`
  - [ ] Link dalla card news alla pagina dettaglio

## Task 5: Aggiornare la documentazione
- **Stato**: todo
- **Scopo**: allineare i doc di progetto alle nuove funzionalità.
- **Risultato atteso**: doc aggiornati e coerenti.
- **Todolist**:
  - [ ] `doc-new-app/06-api-and-webapp.md`: nuovi endpoint e nuova vista news
  - [ ] `doc-new-app/07-operativita-batch.md`: job `cleanup` in cron
  - [ ] `doc-new-app/04-news-llm-pipeline.md`: nota su retention 14 giorni
