# Piano Operativo: Flag "Analisi Notizie LLM" (NEWS_LLM_ENABLED)

Questo documento definisce il piano per aggiungere un flag di configurazione che
consenta di attivare/disattivare l'analisi LLM (DeepSeek) delle notizie, mantenendo
sempre attivo il solo download da yfinance. Utile durante assenze (es. vacanze) per
non consumare token. Persistenza in `.env` + toggle da API e webapp.

---

## Task 1: Config flag in `app/config.py`

- **Stato**: done
- **Scopo**: Leggere `NEWS_LLM_ENABLED` dall'ambiente (default `true`) ed esporre
  helpers per leggere/aggiornare il valore persistito nel file `.env` (lettura fresh,
  senza cache, così l'API riflette subito il cambio senza restart).
- **Risultato atteso**: `settings.NEWS_LLM_ENABLED` presente; funzioni
  `get_news_llm_enabled()` e `set_news_llm_enabled(value)` in `config.py` che
  sostituiscono/aggiungono la riga `NEWS_LLM_ENABLED=` nel `.env`.
- **Test**: `python -c "from app.config import settings, get_news_llm_enabled; print(settings.NEWS_LLM_ENABLED, get_news_llm_enabled())"`.
- **Todolist**:
  - [x] Aggiungere attributo `NEWS_LLM_ENABLED` in `Settings`.
  - [x] Aggiungere `get_news_llm_enabled()` / `set_news_llm_enabled()` con scrittura idempotente sul `.env`.
  - [x] Verifica lettura dal venv.

## Task 2: Early-exit job `sentiment`

- **Stato**: done
- **Scopo**: Se il flag è disabilitato, il job `sentiment` non chiama DeepSeek.
- **Risultato atteso**: `run_sentiment_job` ritorna `{"analyzed": 0, "status": "disabled"}` con log informativo.
- **Todolist**:
  - [x] Inserire il check su `get_news_llm_enabled()` all'inizio di `run_sentiment_job`.

## Task 3: API endpoints `GET/PUT /api/config/news-llm`

- **Stato**: done
- **Scopo**: Esporre il flag via REST in `api/routers/config.py`.
- **Risultato atteso**: `GET /api/config/news-llm` → `{"news_llm_enabled": bool}`;
  `PUT /api/config/news-llm` con body `{"news_llm_enabled": bool}` validato → scrive `.env` e ritorna il valore aggiornato.
- **Todolist**:
  - [x] Aggiungere endpoint GET.
  - [x] Aggiungere endpoint PUT con validazione del body (pydantic).

## Task 4: `.env.example` e `.env`

- **Stato**: done
- **Scopo**: Documentare la nuova variabile nel template e impostarla nel `.env` locale.
- **Risultato atteso**: `NEWS_LLM_ENABLED=true` presente in `.env.example` e `.env`.
- **Todolist**:
  - [x] Aggiungere `NEWS_LLM_ENABLED=true` a `.env.example`.
  - [x] Aggiungere `NEWS_LLM_ENABLED=true` a `.env` (se assente).

## Task 5: Frontend — ApiService + toggle in Configurazione

- **Stato**: done
- **Scopo**: Aggiungere i metodi API e un toggle "Analisi notizie IA" nella pagina
  Configurazione (stile Cyber Light HUD).
- **Risultato atteso**: toggle caricato all'init (GET), salva al cambio (PUT) con feedback.
- **Todolist**:
  - [x] `api.service.ts`: `getNewsLlmEnabled()` e `updateNewsLlmEnabled(enabled)`.
  - [x] `config.component.ts`: stato, template e stile del toggle.
  - [x] `npm run build` OK.

## Task 6: Documentazione (04 e 07)

- **Stato**: done
- **Scopo**: Allineare i documenti di progetto in italiano.
- **Risultato atteso**: `07-operativita-batch.md` §2 con la variabile e spiegazione;
  `04-news-llm-pipeline.md` con la nota sul flag.
- **Todolist**:
  - [x] Aggiornare `doc-new-app/07-operativita-batch.md` (sezione 2).
  - [x] Aggiornare `doc-new-app/04-news-llm-pipeline.md` (sezione 2 o nuova nota).

## Task 7: Verifica finale

- **Stato**: done
- **Scopo**: Smoke test end-to-end del flag.
- **Risultato atteso**: nessun errore; flag OFF → `sentiment` in `disabled`; build web OK.
- **Todolist**:
  - [x] `python -m compileall` sui moduli modificati.
  - [x] Smoke test `python -m app.cli sentiment` con flag OFF (status disabled) e ON (comportamento normale).
  - [x] Test GET/PUT `/api/config/news-llm`.
  - [x] `npm run build` in `hayai-new/web`.
