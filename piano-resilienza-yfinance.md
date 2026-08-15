# Piano Operativo: Resilienza yfinance contro il Rate Limit 429

Questo documento definisce il piano per rendere i job batch notturni di HAYAI v2
resilienti al rate limit di Yahoo Finance (HTTP 429 Too Many Requests), che blocca
periodicamente l'IP del Raspberry Pi.

---

## Task 1: Helper condiviso con retry/backoff per yfinance
- **Stato**: done
- **Scopo**: Creare un modulo riutilizzabile che incapsula una sessione yfinance
  con User-Agent valido, retry esponenziale con jitter su errori 429/5xx e JSON
  malformati, e delay "polite" tra le richieste.
- **Risultato atteso**: Nuovo file `hayai-new/app/yf_client.py` con una classe
  (es. `YahooFinanceClient`) che espone `download_history`, `fetch_info` e
  `fetch_news` con retry automatico. Nessun comportamento rotto sui job esistenti.
- **Test**: Import del modulo senza errori; chiamata a un simbolo con rete attiva
  e simulazione di retry con un endpoint fittizio che fallisce.
- **Todolist**:
  - [x] Creare `hayai-new/app/yf_client.py` con sessione `requests.Session` condivisa
  - [x] Implementare retry con backoff esponenziale + jitter su 429/5xx/JSON error
  - [x] Esporre metodi `download_history`, `fetch_info`, `fetch_news`
  - [x] Verificare l'import del modulo nel venv

## Task 2: Refactor job `data` per usare il client resilient
- **Stato**: done
- **Scopo**: Sostituire la creazione manuale della sessione e la gestione errori
  nel job `data` con il nuovo client, così i 429 vengono ritentati invece di
  produrre "No history returned".
- **Risultato atteso**: `hayai-new/app/jobs/data.py` usa `YahooFinanceClient`
  per scaricare gli OHLCV con retry automatico.
- **Test**: Esecuzione manuale del job `data` sul PC (rete attiva) o almeno
  verifica di sintassi/import.
- **Todolist**:
  - [x] Integrare `YahooFinanceClient` in `data.py`
  - [x] Rimuovere la sessione manuale e il delay inline duplicato
  - [x] Verificare import e sintassi

## Task 3: Refactor job `metadata` per usare il client resilient
- **Stato**: done
- **Scopo**: Sostituire la gestione errori del job `metadata` (che oggi fallisce
  con "429 Too Many Requests" su `ticker.info`) con retry/backoff del client.
- **Risultato atteso**: `hayai-new/app/jobs/metadata.py` usa `YahooFinanceClient`
  e ritenta automaticamente i 429; gli errori residuali vengono loggati senza
  far fallire l'intero ciclo.
- **Test**: Import/sintassi; se possibile run manuale del job `metadata`.
- **Todolist**:
  - [x] Integrare `YahooFinanceClient` in `metadata.py`
  - [x] Rimuovere la sessione manuale e il delay inline duplicato
  - [x] Verificare import e sintassi

## Task 4: Refactor job `news` per usare il client resilient
- **Stato**: done
- **Scopo**: Applicare lo stesso pattern al job `news` (che usa `ticker.news`),
  così da ritentare anche le richieste notizie soggette a rate limit.
- **Risultato atteso**: `hayai-new/app/jobs/news.py` usa `YahooFinanceClient`.
- **Test**: Import/sintassi.
- **Todolist**:
  - [x] Integrare `YahooFinanceClient` in `news.py`
  - [x] Verificare import e sintassi

## Task 5: Aggiornamento documentazione operativa
- **Stato**: done
- **Scopo**: Documentare il comportamento resiliente al rate limit nel manuale di
  manutenzione, in modo che la procedura di troubleshooting rifletta il nuovo
  comportamento (retry automatico).
- **Risultato atteso**: Voce aggiornata in `doc-new-app/11-maintenance-manual.md`
  sulla gestione del rate limit yfinance.
- **Todolist**:
  - [x] Aggiornare la riga sul rate limit in `11-maintenance-manual.md`
