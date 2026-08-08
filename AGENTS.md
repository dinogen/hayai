# HAYAI v2 — Istruzioni per agenti AI (Nuova Applicazione)

## 1. Chi siamo e Obiettivo

**HAYAI v2** è un sistema di supporto decisionale e portafoglio quantitativo personale
(“Personal Quant”) basato su un esperimento da **€5.000**, progettato per girare su un
**Raspberry Pi** come webservice.

Il progetto viene **scritto da zero** (pulito, senza riutilizzare il codice legacy nella
root, che serve solo come archivio storico). Le specifiche complete e definitive si
trovano nella cartella **`doc-new-app/`**.

## 2. Decisioni Architetturali Chiave (Vincolanti)

| Elemento | Scelta |
|---|---|
| **Portafoglio** | Unico portafoglio personale (capitale iniziale €5.000, 90% investito, 10% cash). |
| **Asset Class** | Azioni, ETF e Rendimenti Obbligazionari (es. `^TNX`). **Nessun Forex**. |
| **Fonti Dati & Notizie** | **yfinance** (prezzi OHLCV, indici, notizie). **Alpaca eliminato**. |
| **Modello Quant** | Rete Keras MLP (addestrata periodicamente su PC in Jupyter, esportata in ONNX). |
| **Intelligenza Artificiale** | **DeepSeek API** per analizzare le notizie, estrarre sentiment e generare la tesi di investimento (`rationale`) in italiano. |
| **Segnale** | **Ibrido**: `final_signal = quant_score + llm_sentiment_modifier`. |
| **Esecuzione** | **Nessun broker / nessun ordine automatico**: il sistema produce raccomandazioni long/short e schede tesi per la revisione umana del **martedì** con il promotore finanziario. |
| **Database** | **MariaDB** sul Raspberry Pi. |
| **Backend** | **FastAPI** (REST in sola lettura). |
| **Frontend** | **Angular SPA** (servita da nginx). |
| **Batch** | Script Python CLI schedulati via **cron** (installazione nativa su Raspberry Pi). |

## 3. Convenzioni

| Elemento | Regola |
|---|---|
| **Documentazione** | in **italiano** (nella cartella `doc-new-app/`) |
| **Codice sorgente** | in **inglese** |
| Nomi file documenti | in **inglese** (kebab-case) |

## 4. Struttura della Documentazione di Progetto (`doc-new-app/`)

Consulta SEMPRE questi documenti prima di scrivere o modificare qualsiasi codice del nuovo progetto:

| File | Contenuto |
|---|---|
| `00-index.md` | Indice generale e filosofia di HAYAI v2 |
| `01-target-architecture.md` | Panoramica architetturale, componenti e flusso operativo |
| `02-database-schema.md` | DDL MariaDB completo (tabelle, indici, NAV cash/posizioni) |
| `03-ml-pipeline.md` | Training su PC (Jupyter) → ONNX → inferenza batch sul Pi |
| `04-news-llm-pipeline.md` | Ingestione notizie yfinance + prompt strutturato JSON per DeepSeek API |
| `05-portfolio-optimization.md` | Selezione top long / bottom short, allocazione importi su €5.000 |
| `06-api-e-webapp.md` | API FastAPI REST + Viste Angular (Schede Tesi di Investimento) |
| `07-operativita-batch.md` | CLI batch, pianificazione cron notturna, deploy nativo Raspberry Pi |
| `08-portfolio-lifecycle.md` | Giorno 1 (bootstrap), Mark-to-Market giornaliero, gestione universo |
| `09-ui-ux-design-system.md` | Design System "Cyber Light HUD" (temi, font futuristiche, card tesi) |

## 5. Regole Operative per gli Agenti

1. **Non toccare il codice legacy nella root**: serve solo come archivio concettuale. Tutto il codice nuovo va scritto da zero nella struttura definita in `doc-new-app/`.
2. **Consulta `doc-new-app/` prima di agire**: ogni dubbio su tabelle, flussi o API trova risposta nei documenti di progetto.
3. **Rispetta le convenzioni**: codice in inglese, documentazione e commenti complessi in italiano (dove richiesto).
4. **Chiedi in caso di ambiguità**: se un requisito non è chiaro nei documenti, fermati e chiedi prima di implementare.
5. **Ogni volta che crei un piano operativo** (lavoro multi-task), carica la skill `crea-piano` (`.opencode/skills/crea-piano/SKILL.md`): il piano va salvato in un file che inizia per `piano` e ogni task numerato deve avere stato, scopo, risultato atteso e todolist (modello di riferimento: `piano-training-modello.md`).
6. **Installa sempre i moduli mancanti**: puoi installare liberamente i moduli Python nel `venv` (`venv\Scripts\python -m pip install ...`) e i moduli Node.js in `hayai-new\web` (`npm.cmd install ...`). Se qualcosa ti manca, installalo invece di cercare workaround o bypass; in caso di dubbio sulla scelta del modulo, chiedi all'utente.
7. **Shell disponibili**: puoi usare PowerShell, `bash`, `node` e `python`. Se PowerShell è limitato (es. rendering UTF-8, comandi Unix), usa il Git Bash di sistema: `C:\Users\semboli\AppData\Local\Programs\Git\git-bash.exe` (o `bash.exe` in `C:\Users\semboli\AppData\Local\Programs\Git\bin\`).
