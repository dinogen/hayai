# 07 — Operatività: Job Batch, Cron e Deploy su Raspberry Pi

Questo documento descrive l'esecuzione dei **job batch notturni**, la configurazione
delle **variabili d'ambiente (DeepSeek API key)**, la pianificazione **cron** e la
guida passo-passo per il **deploy nativo su Raspberry Pi**.

---

## 1. Struttura dei Job Batch CLI

I processi batch sono invocati tramite un'interfaccia a riga di comando unificata:
```bash
python -m app.cli <job_name> [--portfolio <code>]
```

### Elenco dei Job Notturni
1. **`data`**: Scarica i prezzi giornalieri OHLCV, forex e indici da yfinance e fa l'upsert in `price_daily`, `fx_rate`, `index_value`. Usa una cache parquet (TTL 24h) per evitare richieste ripetute; con il flag `--refresh` salta la cache e riscarica gli ultimi 5 giorni (usato per gli aggiornamenti intraday, vedi §3).
2. **`metadata`**: Scarica settore, country e area degli strumenti (`sector`/`category`, `country`) da yfinance e aggiorna `instrument`. Salta gli strumenti aggiornati da meno di 30 giorni, oppure forza il refresh con `--force`.
3. **`news`**: Scarica le notizie recenti per tutti gli strumenti attivi da yfinance (`Ticker.news`) e le salva in `news`.
4. **`news_rss`**: Scarica le notizie per-azienda da **Google News RSS** (query `"SIMBOLO" OR "nome"`), con **dedup per titolo** contro le notizie già in tabella (finestra retention) per evitare duplicati tra le due sorgenti; upsert in `news`. Inserito tra `news` e `sentiment` così le notizie fresche vengono analizzate nella stessa notte.
5. **`sentiment`**: Invia le nuove notizie alle API di **DeepSeek**, ricava sentiment, confidence e rationale, e popola `news_sentiment`. **Viene saltato se `NEWS_LLM_ENABLED=false`** (vedi §2): in tal caso termina con stato `disabled` senza consumare token.
6. **`predict`**: Esegue l'inferenza ONNX (`model_prediction`) utilizzando i modelli attivi in `model_registry`.
7. **`signal`**: Combina `model_prediction` e `news_sentiment` per calcolare il segnale ibrido in `portfolio_signal`.
8. **`recommend`**: Calcola i pesi finali long/short e popola `portfolio_recommendation`.
9. **`nav`**: Mark-to-Market giornaliero: allinea le posizioni simulate alla raccomandazione e calcola NAV/cash in `portfolio_position` e `portfolio_cash`.
10. **`summaries`**: Compila il riassunto in Markdown per portafoglio e lo salva in `news_summary`.
11. **`cleanup`**: Elimina le notizie (e relative `news_sentiment` in cascata) più vecchie di 14 giorni e i file cache parquet scaduti in `tmp/`. Il periodo di retention è configurabile con `--days` (default 14).

### Job Settimanale (fuori dal ciclo notturno)
12. **`align`**: **Allineamento del portafoglio alle raccomandazioni**, schedulato **una volta a
    settimana, il martedì alle 15:20** (dopo il refresh prezzi intraday delle 15:00, così i trade
    usano prezzi freschi). NON fa parte del ciclo giornaliero.
    - Legge l'ultima `rec_date` da `portfolio_recommendation` e genera i trade necessari per portare
      le posizioni attuali alla composizione target (chiusura posizioni fuori target, apertura/
      incremento/riduzione long e short), registrandoli in `portfolio_trade` e aggiornando
      `portfolio_position` e `portfolio_cash`.
    - **Soglia di tolleranza**: rispetta `rebalance_threshold_eur` (default €50): le variazioni
      same-direction sotto soglia restano invariate (hold), evitando micro-operazioni. Aperture e
      chiusure vengono sempre eseguite.
    - **Guardia anti-stale**: se l'ultima `rec_date` è più vecchia di 4 giorni (default, `--days`)
      il job si ferma con stato `skipped` senza operare; si forza con `--force`.
    - La logica di trade è condivisa con l'endpoint `holdings/save` (`app/portfolio_rebalance.py`).

### Job di Manutenzione (manuale)
- **`universe`**: Semina/aggiorna l'**universo dei candidati** in `instrument` (~100 simboli di
  training da `UNIVERSE_SYMBOLS`) **senza** linkarli al portafoglio (`portfolio_instrument` resta
  invariato). Gli strumenti già presenti vengono saltati (idempotente). Dalla pagina Watchlist della
  webapp si possono poi aggiungere/rimuovere ticker scegliendoli da questo pool. Da lanciare
  manualmente quando si vuole ampliare l'universo disponibile:
  ```bash
  python -m app.cli universe
  ```
  NB: la stessa logica è usata da `train_universe_pipeline.seed_universe()` (training su PC).

### Job di Verifica (manuale)
- **`verify`**: Valuta il modello ML deployato sul dataset attuale (assenza di null/NaN, split 80/20,
  metriche RMSE/MAE/R²/hit-rate, spot check di 100 righe) e produce un report in
  `logs/model_verification_*.txt` (vedi `03-ml-pipeline.md` §5). **Non va messo in cron**: è un
  controllo manuale da lanciare dopo un retraining o in caso di sospetto drift dei dati:
  ```bash
  python -m app.cli verify
  ```

---

## 2. Configurazione e Credenziali (`.env` e `.env.example`)

Tutti i parametri sensibili e le chiavi API risiedono nel file `.env` nella root del progetto (inserito in `.gitignore` e mai versionato). Esiste un file di template versionato chiamato `.env.example` da cui partire:

```env
# Database MariaDB
DB_HOST=127.0.0.1
DB_PORT=3306
DB_NAME=hayai
DB_USER=hayai
DB_PASSWORD=tua_password_sicura

# DeepSeek API
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_API_BASE_URL=https://api.deepseek.com/v1

# Analisi notizie IA (DeepSeek)
# false = il job 'sentiment' salta l'analisi LLM (le notizie vengono comunque
# scaricate dal job 'news'), utile durante le assenze per non consumare token.
# Il flag è modificabile anche da webapp (Configurazione -> Analisi Notizie IA).
NEWS_LLM_ENABLED=true

# FastAPI / Uvicorn
API_HOST=127.0.0.1
API_PORT=8000

# Autenticazione (un solo utente, sessione via cookie)
# AUTH_SESSION_SECRET: stringa casuale lunga, generata con:
#   python -c "import secrets; print(secrets.token_hex(32))"
AUTH_USERNAME=il_tuo_utente
AUTH_PASSWORD=la_tua_password
AUTH_SESSION_SECRET=esadecimale-casuale-lungo
AUTH_SESSION_MAX_AGE=43200
```

---

## 3. Pianificazione Cron (Notturna)

Crontab dell'utente di sistema (`dinogen`) sul Raspberry Pi per l'esecuzione automatica notturna:

```cron
# Esecuzione sequenziale notturna (Lun-Ven alle 02:15)
15 2 * * 1-5   cd /opt/hayai/hayai-new && /opt/hayai/venv/bin/python -m app.cli data >> logs/cron.log 2>&1
30 2 * * 1-5   cd /opt/hayai/hayai-new && /opt/hayai/venv/bin/python -m app.cli metadata >> logs/cron.log 2>&1
45 2 * * 1-5   cd /opt/hayai/hayai-new && /opt/hayai/venv/bin/python -m app.cli news >> logs/cron.log 2>&1
00 3 * * 1-5   cd /opt/hayai/hayai-new && /opt/hayai/venv/bin/python -m app.cli sentiment >> logs/cron.log 2>&1
15 3 * * 1-5   cd /opt/hayai/hayai-new && /opt/hayai/venv/bin/python -m app.cli predict >> logs/cron.log 2>&1
30 3 * * 1-5   cd /opt/hayai/hayai-new && /opt/hayai/venv/bin/python -m app.cli signal >> logs/cron.log 2>&1
45 3 * * 1-5   cd /opt/hayai/hayai-new && /opt/hayai/venv/bin/python -m app.cli recommend >> logs/cron.log 2>&1
50 3 * * 1-5   cd /opt/hayai/hayai-new && /opt/hayai/venv/bin/python -m app.cli nav >> logs/cron.log 2>&1
00 4 * * 1-5   cd /opt/hayai/hayai-new && /opt/hayai/venv/bin/python -m app.cli summaries >> logs/cron.log 2>&1
30 4 * * 1-5   cd /opt/hayai/hayai-new && /opt/hayai/venv/bin/python -m app.cli cleanup --days 14 >> logs/cron.log 2>&1

# Backup giornaliero del database alle 04:15
0  4 * * *     cd /opt/hayai/hayai-new && scripts/backup.sh >> logs/backup.log 2>&1

# Aggiornamento prezzi intraday (refresh forzato, salta la cache):
# ogni ora durante l'orario del mercato US (es. 15:30–22:00 CET).
# 'data' usa la cache parquet per 24h; --refresh la ignora e riscarica gli
# ultimi 5 giorni da yfinance. Regola le ore in base al tuo fuso orario.
0  15-21 * * 1-5   cd /opt/hayai/hayai-new && /opt/hayai/venv/bin/python -m app.cli data --refresh >> logs/cron.log 2>&1

# Mark-to-market intraday: rivaluta il NAV coi prezzi aggiornati subito dopo il refresh
5  15-21 * * 1-5   cd /opt/hayai/hayai-new && /opt/hayai/venv/bin/python -m app.cli nav >> logs/cron.log 2>&1

# Allineamento settimanale del portafoglio alle raccomandazioni (martedì, NON parte
# del ciclo giornaliero). Schedulato dopo il refresh intraday delle 15:00 (minuto 20)
# per eseguire i trade ai prezzi appena rinfrescati. Rispetta rebalance_threshold_eur
# e salta raccomandazioni stale (> 4 giorni) a meno di --force.
20 15 * * 2      cd /opt/hayai/hayai-new && /opt/hayai/venv/bin/python -m app.cli align >> logs/cron.log 2>&1
```

---

## 4. Guida al Deploy Nativo su Raspberry Pi

### 1. Sistema Operativo e MariaDB
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y mariadb-server nginx python3-venv python3-pip build-essential
sudo mysql_secure_installation
sudo mariadb -e "CREATE DATABASE hayai CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
sudo mariadb -e "CREATE USER 'hayai'@'localhost' IDENTIFIED BY 'tua_password_sicura';"
sudo mariadb -e "GRANT ALL PRIVILEGES ON hayai.* TO 'hayai'@'localhost'; FLUSH PRIVILEGES;"
```

### 2. Setup Ambiente Python
```bash
sudo mkdir -p /opt/hayai/hayai-new && sudo chown $USER /opt/hayai
python3 -m venv /opt/hayai/venv
/opt/hayai/venv/bin/pip install --upgrade pip
/opt/hayai/venv/bin/pip install -r requirements.txt
```

### 3. Configurazione Servizio Systemd (FastAPI)
Il file di unit è versionato nel repo in `deploy/hayai-api.service` e si installa con
`scripts/install_api_service.sh` (abbrevia i comandi seguenti). In alternativa, crea
`/etc/systemd/system/hayai-api.service` manualmente:

```ini
[Unit]
Description=HAYAI v2 FastAPI Service
Wants=network-online.target
After=mariadb.service network-online.target

[Service]
Type=simple
User=dinogen
WorkingDirectory=/opt/hayai/hayai-new
EnvironmentFile=/opt/hayai/hayai-new/.env
ExecStart=/opt/hayai/venv/bin/uvicorn api.main:app --host 127.0.0.1 --port 8000
Restart=on-failure
RestartSec=5
StandardOutput=append:/opt/hayai/hayai-new/logs/api.log
StandardError=append:/opt/hayai/hayai-new/logs/api.log

[Install]
WantedBy=multi-user.target
```
Abilita e avvia (o usa lo script):
```bash
sudo scripts/install_api_service.sh
# oppure manualmente:
sudo systemctl daemon-reload
sudo systemctl enable --now hayai-api
```
Verifica che parta al boot:
```bash
systemctl is-enabled hayai-api     # atteso: enabled
curl http://127.0.0.1:8000/api/health
```

### 4. Configurazione Nginx (Frontend + Reverse Proxy API)
Il sito è versionato nel repo in `deploy/nginx-hayai.conf` e si deploya con
`scripts/deploy_web.sh` (copia la build Angular in `/var/www/hayai`, installa e
attiva il sito, rimuove il default, `nginx -t` e reload). Config manuale
equivalente — crea `/etc/nginx/sites-available/hayai`:
```nginx
server {
    listen 80;
    server_name _;
    root /var/www/hayai;
    index index.html;

    location /api/ {
        proxy_pass http://127.0.0.1:8000/api/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
    location / {
        try_files $uri $uri/ /index.html;
    }
}
```
Deploy in un comando (dalla root `/opt/hayai/hayai-new`):
```bash
sudo scripts/deploy_web.sh
```
Oppure manualmente:
```bash
sudo cp -r web/dist/web/browser/. /var/www/hayai/
sudo ln -s /etc/nginx/sites-available/hayai /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx
```

> **Nota sulla build**: in produzione il frontend chiama l'API in **relativo** (`/api`),
> quindi funziona da qualsiasi browser. In sviluppo (senza nginx) resta
> `http://127.0.0.1:8000/api` tramite gli environment Angular
> (`src/environments/environment.ts` ↔ `environment.prod.ts`).
