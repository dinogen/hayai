# Piano Operativo: Login a Sessione per HAYAI v2

Questo documento definisce il piano per aggiungere un sistema di autenticazione
semplice (un solo utente) alla webapp HAYAI v2: credenziali fisse in `.env`,
sessione via cookie firmato (nessun JWT, nessuno store lato server), API
protette fuori dalla sessione.

**Scelte chiave**:
- Sessione: `SessionMiddleware` di Starlette (cookie firmato, `itsdangerous`).
- Credenziali: `AUTH_USERNAME` / `AUTH_PASSWORD` in `.env` (confronto con `hmac.compare_digest`).
- Durata sessione: 12 ore (`AUTH_SESSION_MAX_AGE=43200`).
- Pubblici: `/api/auth/login` e `/api/health`. Tutte le altre API richiedono sessione.
- Dev cross-origin: CORS a origini esplicite + `withCredentials` nel frontend.

---

## Task 1: Pianificazione e credenziali in `.env`
- **Stato**: done
- **Scopo**: rendere configurabili utente, password e segreto di firma del cookie di sessione.
- **Risultato atteso**: `.env.example` versionato con le nuove chiavi e `.env` locale aggiornato con valori reali (username, password, secret casuale).
- **Todolist**:
  - [x] Aggiungere `AUTH_USERNAME`, `AUTH_PASSWORD`, `AUTH_SESSION_SECRET`, `AUTH_SESSION_MAX_AGE` a `.env.example`
  - [x] Aggiornare `.env` locale (secret generato con `secrets.token_hex(32)`)

## Task 2: Dipendenza `itsdangerous`
- **Stato**: done
- **Scopo**: installare la dipendenza richiesta da `SessionMiddleware` di Starlette.
- **Risultato atteso**: `requirements.txt` aggiornato e pacchetto installato nel venv (`python -c "import itsdangerous"` ok).
- **Todolist**:
  - [x] Aggiungere `itsdangerous==2.2.0` a `requirements.txt`
  - [x] `venv\Scripts\python -m pip install itsdangerous==2.2.0`
  - [x] Verifica import nel venv

## Task 3: Config auth nel backend
- **Stato**: done
- **Scopo**: esporre i parametri auth tramite `Settings` e fallire allo startup se mancano.
- **Risultato atteso**: `app/config.py` con `AUTH_USERNAME`, `AUTH_PASSWORD`, `AUTH_SESSION_SECRET`, `AUTH_SESSION_MAX_AGE` e messaggio d'errore chiaro se assenti.
- **Todolist**:
  - [x] Aggiungere campi auth a `Settings`
  - [x] Aggiungere validazione all'import del modulo

## Task 4: Modulo auth API (`api/auth.py`)
- **Stato**: done
- **Scopo**: definire la dipendenza `require_auth` e i parametri del cookie di sessione.
- **Risultato atteso**: `api/auth.py` con `SESSION_KEY`, `require_auth` (401 se non autenticato), helper `is_authenticated`.
- **Todolist**:
  - [x] Creare `api/auth.py`

## Task 5: Router autenticazione (`api/routers/auth.py`)
- **Stato**: done
- **Scopo**: esporre login, logout e stato sessione.
- **Risultato atteso**: `POST /api/auth/login`, `POST /api/auth/logout`, `GET /api/auth/me` funzionanti.
- **Todolist**:
  - [x] Creare `api/routers/auth.py`

## Task 6: Integrazione in `api/main.py`
- **Stato**: done
- **Scopo**: registrare il middleware di sessione, sistemare CORS, proteggere i router business.
- **Risultato atteso**: app FastAPI con sessione attiva, CORS con origini esplicite, `Depends(require_auth)` sui 5 router business; `/api/health` e `/api/auth/*` pubblici.
- **Todolist**:
  - [x] Registrare `SessionMiddleware`
  - [x] Aggiornare CORS (origini esplicite + credentials)
  - [x] Applicare `require_auth` ai router business
  - [x] Includere il router auth

## Task 7: Auth service + interceptor frontend
- **Stato**: done
- **Scopo**: gestire login/logout/stato autenticazione e inviare i cookie nelle request.
- **Risultato atteso**: `auth.service.ts` con signal `authenticated`; interceptor che aggiunge `withCredentials: true`; `app.config.ts` con `provideHttpClient(withInterceptors(...))`.
- **Todolist**:
  - [x] Creare `core/services/auth.service.ts`
  - [x] Creare `core/interceptors/credentials.interceptor.ts`
  - [x] Aggiornare `app.config.ts`

## Task 8: Guard, pagina login e routes
- **Stato**: done
- **Scopo**: bloccare le rotte non autenticate e fornire la pagina di login.
- **Risultato atteso**: `auth.guard.ts` (redirect a `/login`), `login.component.ts` (stile Cyber Light HUD), rotte aggiornate con `canActivate` e rotta `/login`.
- **Todolist**:
  - [x] Creare `core/guards/auth.guard.ts`
  - [x] Creare `features/login/login.component.ts`
  - [x] Aggiornare `app.routes.ts`

## Task 9: Navbar logout + gestione 401
- **Stato**: done
- **Scopo**: mostrare "Esci" quando autenticato e far scadere la sessione sul 401.
- **Risultato atteso**: pulsante logout nella navbar; su 401 da API business → stato non autenticato e redirect a `/login` (senza loop su `/auth/me`).
- **Todolist**:
  - [x] Aggiornare `navbar.component.ts`
  - [x] Gestione 401 nell'AuthService/ApiService

## Task 10: Documentazione
- **Stato**: done
- **Scopo**: aggiornare i documenti di progetto con l'autenticazione.
- **Risultato atteso**: `doc-new-app/06-api-and-webapp.md` con sezione auth; `doc-new-app/07-operativita-batch.md` con template `.env` aggiornato.
- **Todolist**:
  - [x] Aggiornare `06-api-and-webapp.md`
  - [x] Aggiornare `07-operativita-batch.md`

## Task 11: Verifica finale
- **Stato**: done
- **Scopo**: verificare il comportamento end-to-end.
- **Risultato atteso**: backend con health pubblico e API business 401 senza cookie, login/logout funzionanti; build Angular senza errori (`ng build`).
- **Todolist**:
  - [x] Avviare uvicorn e testare con curl (health, 401, login, cookie, logout)
  - [x] `npm run build` nel frontend

## Note operative (fix rilevato in fase di test)
- **Problema**: in dev il login sembrava bloccato. Causa: il cookie `SameSite=Lax`
  non viene inviato dal browser su richieste cross-site (`localhost:4200` →
  `127.0.0.1:8000`), quindi dopo il login la guard di rotta rimbalzava su `/login`.
- **Soluzione**: proxy di sviluppo Angular (`web/proxy.conf.json` abilitato in
  `angular.json` via `proxyConfig`) + `apiUrl` relativo (`/api`) anche in dev.
  Tutte le richieste diventano stessa origine e il cookie di sessione funziona
  come in produzione (nginx).
- **Credenziali**: dopo la modifica dell'utente, `.env` locale usa
  `AUTH_USERNAME=admin` / `AUTH_PASSWORD=abc123` (esempio locale; in produzione
  impostare una password robusta e rigenerare `AUTH_SESSION_SECRET`).

