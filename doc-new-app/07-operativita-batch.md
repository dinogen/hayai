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
1. **`data`**: Scarica i prezzi giornalieri OHLCV, forex e indici da yfinance e fa l'upsert in `price_daily`, `fx_rate`, `index_value`.
2. **`metadata`**: Scarica settore, country e area degli strumenti (`sector`/`category`, `country`) da yfinance e aggiorna `instrument`. Salta gli strumenti aggiornati da meno di 30 giorni, oppure forza il refresh con `--force`.
3. **`news`**: Scarica le notizie recenti per tutti gli strumenti attivi e le salva in `news`.
4. **`sentiment`**: Invia le nuove notizie alle API di **DeepSeek**, ricava sentiment, confidence e rationale, e popola `news_sentiment`. **Viene saltato se `NEWS_LLM_ENABLED=false`** (vedi §2): in tal caso termina con stato `disabled` senza consumare token.
5. **`predict`**: Esegue l'inferenza ONNX (`model_prediction`) utilizzando i modelli attivi in `model_registry`.
6. **`signal`**: Combina `model_prediction` e `news_sentiment` per calcolare il segnale ibrido in `portfolio_signal`.
7. **`recommend`**: Calcola i pesi finali long/short e popola `portfolio_recommendation`.
8. **`nav`**: Mark-to-Market giornaliero: allinea le posizioni simulate alla raccomandazione e calcola NAV/cash in `portfolio_position` e `portfolio_cash`.
9. **`summaries`**: Compila il riassunto in Markdown per portafoglio e lo salva in `news_summary`.
10. **`cleanup`**: Elimina le notizie (e relative `news_sentiment` in cascata) più vecchie di 14 giorni e i file cache parquet scaduti in `tmp/`. Il periodo di retention è configurabile con `--days` (default 14).

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
```

---

## 3. Pianificazione Cron (Notturna)

Crontab dell'utente `hayai` sul Raspberry Pi per l'esecuzione automatica notturna:

```cron
# Esecuzione sequenziale notturna (Lun-Ven alle 02:15)
15 2 * * 1-5   cd /opt/hayai-new && venv/bin/python -m app.cli data >> logs/cron.log 2>&1
30 2 * * 1-5   cd /opt/hayai-new && venv/bin/python -m app.cli metadata >> logs/cron.log 2>&1
45 2 * * 1-5   cd /opt/hayai-new && venv/bin/python -m app.cli news >> logs/cron.log 2>&1
00 3 * * 1-5   cd /opt/hayai-new && venv/bin/python -m app.cli sentiment >> logs/cron.log 2>&1
15 3 * * 1-5   cd /opt/hayai-new && venv/bin/python -m app.cli predict >> logs/cron.log 2>&1
30 3 * * 1-5   cd /opt/hayai-new && venv/bin/python -m app.cli signal >> logs/cron.log 2>&1
45 3 * * 1-5   cd /opt/hayai-new && venv/bin/python -m app.cli recommend >> logs/cron.log 2>&1
50 3 * * 1-5   cd /opt/hayai-new && venv/bin/python -m app.cli nav >> logs/cron.log 2>&1
00 4 * * 1-5   cd /opt/hayai-new && venv/bin/python -m app.cli summaries >> logs/cron.log 2>&1
30 4 * * 1-5   cd /opt/hayai-new && venv/bin/python -m app.cli cleanup --days 14 >> logs/cron.log 2>&1

# Backup giornaliero del database alle 04:15
0  4 * * *     cd /opt/hayai-new && scripts/backup.sh >> logs/backup.log 2>&1
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
sudo mkdir -p /opt/hayai-new && sudo chown $USER /opt/hayai-new
python3 -m venv /opt/hayai-new/venv
/opt/hayai-new/venv/bin/pip install --upgrade pip
/opt/hayai-new/venv/bin/pip install -r requirements.txt
```

### 3. Configurazione Servizio Systemd (FastAPI)
Crea `/etc/systemd/system/hayai-api.service`:
```ini
[Unit]
Description=HAYAI v2 FastAPI Service
After=mariadb.service network-online.target

[Service]
User=hayai
WorkingDirectory=/opt/hayai-new
EnvironmentFile=/opt/hayai-new/.env
ExecStart=/opt/hayai-new/venv/bin/uvicorn api.main:app --host 127.0.0.1 --port 8000
Restart=on-failure

[Install]
WantedBy=multi-user.target
```
Abilita e avvia:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now hayai-api
```

### 4. Configurazione Nginx (Frontend + Reverse Proxy API)
Crea `/etc/nginx/sites-available/hayai`:
```nginx
server {
    listen 80;
    server_name _;
    root /var/www/hayai;
    index index.html;

    location /api/ {
        proxy_pass http://127.0.0.1:8000/api/;
        proxy_set_header Host $host;
    }
    location / {
        try_files $uri $uri/ /index.html;
    }
}
```
Attiva il sito:
```bash
sudo ln -s /etc/nginx/sites-available/hayai /etc/nginx/sites-enabled/
sudo rm /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx
```
*(Copia i file della build di Angular in `/var/www/hayai`)*.
