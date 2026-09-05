# Piano Operativo: Revisione Gestione Notizie — Segmentazione Segnali e Ingestione RSS

Questo documento definisce il piano per correggere la visualizzazione delle notizie nella
pagina Segnali (stesse notizie macro ripetute sotto quasi tutti i ticker) e per migliorare
la copertura delle notizie specifiche di ogni ticker.

**Scoperta chiave dell'analisi**: 20 strumenti su 22 del portafoglio hanno `area='usa'`; il job
`signal` propaga ogni notizia macro (per `impact_surface`) a TUTTI gli strumenti dell'area e la
scrive per intero nel `sentiment_breakdown`. Di conseguenza quasi ogni riga della pagina Segnali
mostra lo stesso identico elenco di notizie macro. Inoltre `news.source_id` è UNIQUE globale
("first-writer-wins"), e il feed `yfinance Ticker.news` espone solo ~10 top-stories generiche,
quindi le notizie realmente specifiche dell'azienda sono poche.

Decisioni concordate:
- **Ambito**: B (segmentazione display signals: notizie dirette / di settore / macro collassate) + C
  (abilitare il job Google News RSS `news_rss`, già scritto ma non collegato). NIENTE refactoring
  dello schema (opzione A, mol-ti-a-molti) in questa iterazione.
- Sul dettaglio segnale le notizie **macro/area** non si ripetono per ticker: un unico blocco
  collassato "Macro {area} (N)". Le **dirette del ticker** sono elencate per prime, seguite dal
  livello **settore GICS** collassato.
- `llm_sentiment_modifier` e `final_signal` **non cambiano**: le macro continuano a contribuire
  al segnale; cambia solo il modo in cui il dettaglio notizie viene presentato.

Limite noto (fuori scope): l'attribuzione "first-writer-wins" degli articoli condivisi resta;
il job RSS per-azienda la mitiga fornendo a ogni ticker articoli propri.

---

## Task 1: API segnali — esporre `area` e `sector` per strumento
- **Stato**: done
- **Scopo**: la pagina Segnali deve conoscere area e settore GICS dello strumento per segmentare il dettaglio notizie.
- **Risultato atteso**: `GET /portfolios/{code}/signals` restituisce per ogni riga anche `i.area` e `i.sector`.
- **Test**: chiamata all'endpoint (curl) o ispezione risposta → campi `area`/`sector` presenti (stringa o null).
- **Todolist**:
  - [x] `api/routers/portfolios.py` (endpoint `signals`): aggiungere `i.area, i.sector` alla SELECT del join `instrument`
  - [x] Verificare che il parsing JSON esistente non si rompa

## Task 2: Frontend signals — dettaglio notizie in 3 livelli di rilevanza
- **Stato**: done
- **Scopo**: nel dettaglio espandibile di ogni segnale separare notizie *dirette del ticker*
  (elencate), *di settore GICS* (collassate) e *macro/area* (un unico blocco collassato), senza
  ripetere visivamente le macro sotto ogni ticker. Non modifica i valori di segnale.
- **Risultato atteso**: espandendo una riga si vedono prima le notizie proprie del ticker, poi i
  blocchi "Settore {sector} (N)" e "Macro {area} (N)" espandibili; l'header della riga riporta
  "X dirette · Y settore · Z macro".
- **Test**: `npm run build` in `hayai-new/web` senza errori; verifica manuale su dati reali.
- **Todolist**:
  - [x] `signals.component.ts`: classificare il breakdown tramite `b.direct` (già nel payload)
  - [x] Tier dirette: `b.direct !== false` dello strumento, ordinate per |contribution|
  - [x] Tier settore: unione dedup (per titolo) delle dirette dei segnali con stesso `sector` (escluso il ticker), blocco collassabile con conteggio e simbolo sorgente
  - [x] Tier macro/area: `b.direct === false` raggruppate in blocco collassabile "Macro {area} (N)" con contributo aggregato
  - [x] Header riga aggiornato con i tre conteggi
  - [x] Gestire `sector`/`area` null e dedup tra livelli

## Task 3: Backend — abilitare ingestione notizie Google News RSS per ticker
- **Stato**: done
- **Scopo**: registrare `app/jobs/news_rss.py` (già scritto, non collegato) nel CLI e nel ciclo
  notturno, con dedup anti-rumore per evitare duplicati tra feed yfinance e Google News.
- **Risultato atteso**: job `news_rss` eseguibile via CLI e incluso nel ciclo notturno (dopo
  `news`, prima di `sentiment`) così le notizie RSS fresche vengono analizzate da DeepSeek nella
  stessa notte; nessun duplicato per titolo.
- **Test**: `python -m app.cli news_rss --portfolio main` (log `news_inserted`); verifica righe `news` nuove e assenza di duplicati titolo.
- **Todolist**:
  - [x] `app/cli.py`: registrare `"news_rss"` in `JOBS_MAP`
  - [x] `scripts/run_nightly.sh`: aggiungere `"news_rss"` tra `news` e `sentiment`
  - [x] `app/jobs/news_rss.py`: dedup per titolo esatto già presente in `news` (finestra retention) prima dell'upsert
  - [x] Nessuna modifica per simboli senza risultati (ETF/indici/tassi: accettabile)

## Task 4: Documentazione di progetto
- **Stato**: done
- **Scopo**: allineare `doc-new-app/` alle modifiche.
- **Risultato atteso**: doc coerenti con codice.
- **Todolist**:
  - [x] `04-news-llm-pipeline.md`: due sorgenti (yfinance + Google News RSS per-azienda) + dedup per titolo
  - [x] `06-api-and-webapp.md`: endpoint `/signals` con `area`/`sector`/`direct`; dettaglio segnali segmentato dirette/settore/macro collassate
  - [x] `07-operativita-batch.md`: job `news_rss` nell'ordine notturno
  - [x] `09-ui-ux-design-system.md`: nessun contenuto specifico sul dettaglio segnali (n/a)
