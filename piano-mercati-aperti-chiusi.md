# Piano Operativo: Box Mercati Aperti/Chiusi in Dashboard

Questo documento definisce il piano per aggiungere alla dashboard di HAYAI v2 un box HUD
che mostra lo stato (aperto/chiuso) dei mercati USA, EU e Asia, con stato calcolato
lato FastAPI e auto-aggiornamento nel frontend Angular.

---

## Task 1: Modulo backend `app/market_hours.py`
- **Stato**: done
- **Scopo**: logica di calcolo dello stato di apertura/chiusura dei mercati (codice in inglese).
- **Risultato atteso**: funzione `get_market_status()` che restituisce per ogni mercato
  `{code, name, timezone, local_time, is_open, open_time, close_time, next_open_at, next_close_at}`;
  giorni feriali (lun-ven), week-end → chiuso con prossima apertura al lunedì.
- **Test**: self-test `if __name__ == "__main__"` (stile `app/area.py`).
- **Todolist**:
  - [x] Definire costanti mercati (code, name, tz IANA, orari open/close) con `zoneinfo.ZoneInfo`
  - [x] Implementare calcolo `is_open` su giorni feriali
  - [x] Calcolare `next_open_at` / `next_close_at`
  - [x] Self-test `if __name__ == "__main__"` con casi noti

## Task 2: Endpoint API `GET /api/markets/status`
- **Stato**: done
- **Scopo**: esporre lo stato dei mercati tramite FastAPI (API read-only).
- **Risultato atteso**: router `api/routers/markets.py` registrato in `api/main.py`;
  risposta `{"markets": [...], "generated_at": <UTC ISO>}`.
- **Test**: avviare uvicorn e chiamare `/api/markets/status`.
- **Todolist**:
  - [x] Creare `api/routers/markets.py`
  - [x] Registrare il router in `api/main.py`
  - [x] Aggiungere `tzdata` a `requirements.txt` (serve a `zoneinfo` su Windows; innocuo su Pi)
  - [x] Installare `tzdata` nel venv

## Task 3: Frontend — box nella dashboard
- **Stato**: done
- **Scopo**: card HUD "Mercati Aperti / Chiusi" nella dashboard Angular.
- **Risultato atteso**: nuova card nella griglia HUD con 3 righe mercato: pallino verde
  `#16a34a` (aperto) / rosso `#dc2626` (chiuso), nome, ora locale + orari in JetBrains Mono;
  auto-aggiornamento ogni 60s. Pattern signal conforme a doc 06 §2.0.
- **Test**: `npm.cmd build` in `hayai-new/web`.
- **Todolist**:
  - [x] Aggiungere `getMarketsStatus()` in `api.service.ts`
  - [x] In `dashboard.component.ts`: `markets = signal<any[]>([])`, fetch in `ngOnInit`,
        `setInterval` 60s, `ngOnDestroy` per cleanup
  - [x] Aggiungere la card nel template (stile Cyber Light HUD)

## Task 4: Documentazione
- **Stato**: done
- **Scopo**: aggiornare `doc-new-app/06-api-and-webapp.md` (in italiano).
- **Risultato atteso**: endpoint `/api/markets/status` aggiunto alla tabella API §1;
  box mercati menzionato nella vista Dashboard §2.2.
- **Test**: nessuno (solo testo).
- **Todolist**:
  - [x] Aggiungere riga endpoint nella tabella §1
  - [x] Menzionare il box mercati in §2.2 punto 1
