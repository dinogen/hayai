# 17 — Operatività: job batch, cron e deploy su Raspberry

Questo documento descrive i **job batch**, la loro **pianificazione cron**, la
**struttura del progetto Python** e il **deploy passo-passo** su Raspberry Pi.

## 1. Job batch

Ogni job è un comando CLI indipendente, idempotente e registra l'esito in
`job_run` (doc `13 §2.12`).

Interfaccia unificata (modulo applicativo `app`):

```
python -m app <job> [--portfolio <code>] [--date YYYY-MM-DD] [--dry-run]
```

Esempi:
```
python -m app data
python -m app news --portfolio medium_tech_usa
python -m app features --portfolio eu
python -m app predict --portfolio medium_tech_usa
python -m app recommend --portfolio medium_tech_usa
```

### 1.1 Elenco job

| Job | Funzione | Input → Output | Doc |
|---|---|---|---|
| `data` | Aggiorna prezzi OHLCV, forex, indici (yfinance) | yfinance → `price_daily`, `fx_rate`, `index_value` | `11`, `13` |
| `news` | Scarica notizie per gli strumenti dei portafogli | yfinance → `news` | `15` |
| `summaries` | Genera riassunti markdown per portafoglio/data | `news` → `news_summary` (+ file .md) | `15` |
| `features` | Calcola feature type-agnostic (per strumento/data) | `price_daily` → (transitorio, non persistito) | `14` |
| `predict` | Inferenza modello sull'ultima data | feature + `model_registry` → `prediction` | `14` |
| `recommend` | Calcola pesi long/short e posizioni indicative | `prediction` + parametri → `recommendation` | `14` |

Dipendenze tra job (ordine di esecuzione):

```
data → news → summaries
data → features → predict → recommend
```

Nota: `features` non materializza le feature in DB (ricalcolo a runtime in
`predict`), quindi può essere un job "puro" di validazione/pre-computo oppure
essere fuso dentro `predict`. Nella pianificazione sotto è eseguito insieme a
`predict` per audit e tracciabilità.

### 1.2 Contratto di un job

Ogni job deve:

1. aprire un record in `job_run` (`status='running'`);
2. eseguire la logica in transazioni idempotenti (upsert);
3. aggiornare `job_run` con `status` (`success`/`partial`/`failed`), `exit_code`,
   `details` (conteggi/errori);
4. scrivere un log strutturato (file `logs/<job>_<date>.log`);
5. su errore: retry interni (solo per chiamate esterne), poi terminazione con
   alert opzionale (Telegram).

## 2. Pianificazione cron

Crontab dell'utente dedicato `hayai` (riferimento; orari notturni europei):

```cron
# Aggiornamento dati + predizioni + raccomandazioni (notturno)
15 2 * * 1-5   cd /opt/hayai && venv/bin/python -m app data >> logs/cron.log 2>&1
25 2 * * 1-5   cd /opt/hayai && venv/bin/python -m app news >> logs/cron.log 2>&1
35 2 * * 1-5   cd /opt/hayai && venv/bin/python -m app summaries >> logs/cron.log 2>&1
45 2 * * 1-5   cd /opt/hayai && venv/bin/python -m app predict >> logs/cron.log 2>&1
55 2 * * 1-5   cd /opt/hayai && venv/bin/python -m app recommend >> logs/cron.log 2>&1

# Backup giornaliero
0  3 * * *     cd /opt/hayai && scripts/backup.sh >> logs/backup.log 2>&1

# Pulizia log (più vecchi di 30 giorni)
0  4 * * *     find /opt/hayai/logs -name '*.log' -mtime +30 -delete
```

Osservazioni:

- Esecuzione **giornaliera** (lun-ven) per dati, notizie, predizioni e
  raccomandazioni (RF-10).
- Gli orari scalati (15/25/35/45/55) evitano sovrapposizioni e rispettano i
  limiti di rate di yfinance.
- Il backup via `mysqldump` è quotidiano (RN-09).
- In alternativa ai `&&` per-cron si può usare una piccola orchestrazione
  (es. uno script `run_daily.sh` che esegue i job in sequenza e si ferma se un
  job fallisce). Su POSIX `&&` è valido.

## 3. Struttura del progetto Python

```
hayai-new/
├─ pyproject.toml            (dipendenze, entry point CLI)
├─ app/
│  ├─ __init__.py
│  ├─ cli.py                 (argparse/click: dispatch dei job)
│  ├─ config.py              (lettura .env, path, parametri)
│  ├─ db.py                  (connessione MariaDB, upsert helpers)
│  ├─ logging_setup.py       (logging strutturato + rotazione)
│  ├─ yf.py                  (wrapper yfinance: download batch, retry, news)
│  ├─ jobs/
│  │  ├─ data.py             (job `data`)
│  │  ├─ news.py             (job `news`)
│  │  ├─ summaries.py        (job `summaries`)
│  │  ├─ features.py         (feature engine type-agnostic)
│  │  ├─ predict.py          (job `predict`, onnxruntime)
│  │  └─ recommend.py        (job `recommend`, pesi long/short)
│  ├─ models_registry.py     (lettura/aggiornamento model_registry)
│  └─ alert.py               (notifica Telegram opzionale)
├─ api/
│  ├─ main.py                (FastAPI app, router /api)
│  ├─ routers/               (portfolios, instruments, predictions, ...)
│  └─ schemas/               (Pydantic DTO)
├─ web/                      (progetto Angular separato)
├─ models/                   (artefatti modello: onnx, mins/maxs, config)
├─ data/summaries/           (export markdown)
├─ logs/
├─ scripts/
│  ├─ backup.sh
│  └─ deploy_model.sh        (registra nuovo modello dal PC)
└─ .env                      (credenziali DB, token Telegram — NON versionato)
```

### 3.1 Principi

- **Logica pura testabile**: feature engine, calcolo pesi, dedup notizie come
  funzioni pure (nessun side-effect), testabili senza DB (RN-07).
- **Strato dati separato**: `db.py` incapsula upsert e query.
- **Segreti in `.env`**: DB password, token Telegram. Nessun segreto in repo.
- **Dipendenze**: `yfinance`, `pandas`, `numpy`, `sqlalchemy` (o `pymysql`+SQL),
  `onnxruntime`, `fastapi`, `uvicorn`, `pydantic-settings`.

## 4. Deploy passo-passo (Raspberry Pi)

### 4.1 Prerequisiti

- Raspberry Pi 4 o 5, **Raspberry Pi OS Bookworm 64-bit**.
- Aggiornamento: `sudo apt update && sudo apt upgrade -y`.

### 4.2 MariaDB

```bash
sudo apt install -y mariadb-server
sudo mysql_secure_installation
sudo mariadb -e "CREATE DATABASE hayai CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
sudo mariadb -e "CREATE USER 'hayai'@'localhost' IDENTIFIED BY '<password>';"
sudo mariadb -e "GRANT ALL PRIVILEGES ON hayai.* TO 'hayai'@'localhost'; FLUSH PRIVILEGES;"
# applicare schema: sudo mariadb hayai < schema.sql
```

- MariaDB ascolta solo su `127.0.0.1` (default) — RN-05.

### 4.3 Python venv e dipendenze

```bash
sudo apt install -y python3-venv python3-pip build-essential
sudo mkdir -p /opt/hayai && sudo chown $USER /opt/hayai
python3 -m venv /opt/hayai/venv
/opt/hayai/venv/bin/pip install -U pip
/opt/hayai/venv/bin/pip install -r /opt/hayai/requirements.txt
```

### 4.4 Artefatti modello

- Dal PC: `scripts/deploy_model.sh` copia la cartella `models/<name>/<version>/`
  via `scp`/`rsync` e inserisce la riga in `model_registry` (via CLI dedicata).
- Riferimenti: `model_registry.artifact_path` → `/opt/hayai/models/<name>/<version>/`.

### 4.5 Backend FastAPI (systemd)

File `/etc/systemd/system/hayai-api.service`:

```ini
[Unit]
Description=HAYAI FastAPI backend
After=mariadb.service network-online.target

[Service]
User=hayai
WorkingDirectory=/opt/hayai
EnvironmentFile=/opt/hayai/.env
ExecStart=/opt/hayai/venv/bin/uvicorn api.main:app --host 127.0.0.1 --port 8000
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now hayai-api
sudo systemctl status hayai-api
```

### 4.6 Frontend Angular + nginx

```bash
# build su PC (o CI): ng build --configuration production
sudo apt install -y nginx
sudo mkdir -p /var/www/hayai
sudo cp -r dist/browser/* /var/www/hayai/
```

File `/etc/nginx/sites-available/hayai`:

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

```bash
sudo ln -s /etc/nginx/sites-available/hayai /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

### 4.7 Cron

```bash
sudo useradd -r -m -s /bin/bash hayai   # (se non creato)
crontab -u hayai -e                     # inserire la sezione §2
```

### 4.8 Verifica finale

```bash
# test API
curl http://127.0.0.1:8000/api/health
# test job manuale
cd /opt/hayai && venv/bin/python -m app data
# controllo job_run
mariadb -u hayai -p hayai -e "SELECT job_name,status,duration FROM job_run ORDER BY id DESC LIMIT 10;"
# webapp
curl http://<raspberry-ip>/
```

## 5. Backup e ripristino

- `scripts/backup.sh`: `mysqldump hayai | gzip > backups/hayai_$(date +%F).sql.gz`
  + copia artefatti (`models/`) e `data/summaries/` (rsync verso NAS o PC).
- Ripristino: `gunzip -c <backup> | mariadb hayai`.

## 6. Requisiti soddisfatti

- RF-10/11/12/13 → §1-2 (job data idempotente).
- RF-20/21/22/23 → §1.1 (`news`, `summaries`).
- RF-33/40 → §1.1 (`predict`, `recommend`).
- RF-60/61/62 → §1-1.2 (CLI, `job_run`, alert).
- RN-01/04/06/09 → §1.2, §2, §5.

## 7. Stime di risorse (Raspberry Pi 4/5)

- DB: qualche centinaio di MB per 500 strumenti × 5 anni.
- Batch serale: pochi minuti (download batch yfinance + inferenza ONNX su MLP).
- FastAPI: utilizzo CPU trascurabile in lettura; uvicorn mono-processo sufficiente.
- Riserve: se i portafogli crescessero molto, valutare limite di rate yfinance e
  partizionamento `price_daily` (doc `13 §4`).
