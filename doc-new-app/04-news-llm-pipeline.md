# 04 — Pipeline Notizie & DeepSeek LLM Integration

Questo documento descrive come la nuova applicazione acquisisce le notizie
finanziarie da **yfinance**, le analizza tramite l'API di **DeepSeek** per estrarre
impatto, durata, superficie e tesi di investimento, e genera i riassunti in
markdown per la webapp.

> **Metodo di riferimento**: il principio di lettura delle notizie è descritto in
> `appunti-notizie.md` — *"Non valutare la notizia. Valuta la sorpresa rispetto alle
> attese e il suo potenziale impatto sui prezzi."*

---

## 1. Acquisizione Notizie (`job news`)

Ogni notte, il batch esegue l'ingestione delle notizie:
1. Interroga la tabella `portfolio_instrument` per ottenere tutti i simboli attivi.
2. Per ogni simbolo, richiama `yfinance.Ticker(symbol).news`.
3. Filtra ed effettua l'**upsert** nella tabella `news` usando il campo nativo `id` di yfinance come `source_id` (garantendo l'idempotenza e prevenendo duplicati).

---

## 2. Analisi Semantica con DeepSeek API

> **Flag `NEWS_LLM_ENABLED`**: il job `sentiment` può essere disattivato impostando
> `NEWS_LLM_ENABLED=false` (nel `.env` o tramite la webapp, *Configurazione →
> Analisi Notizie IA*). In quel caso il job termina subito con stato `disabled`,
> **nessun token DeepSeek viene consumato**, ma il job `news` continua a scaricare
> le notizie da yfinance. Le `news_sentiment` già calcolate restano valide e
> continuano a contribuire al segnale con il loro decadimento naturale (vedi §3).
> Quando si riattiva il flag, le notizie scaricate nel frattempo vengono analizzate
> alla successiva esecuzione del job.

Per ciascuna notizia fresca non ancora analizzata, il sistema invia una richiesta alle **API di DeepSeek** (modello `deepseek-chat` o equivalente raccomandato per task testuali e JSON mode).

### 2.1 Prompt Engineering Strutturato
Per evitare risposte ambigue o formati non parsabili, si forza il modello a restituire un **JSON valido**.

**Schema del Prompt inviato a DeepSeek:**
```text
Sei un analista finanziario quantitativo ed esperto di mercati.
Analizza la seguente notizia finanziaria relativa allo strumento {symbol} ({instrument_name}),
che appartiene all'area {area}.

Titolo: {title}
Editore: {publisher}
Testo/Estratto: {summary}

Metodo: NON valutare la notizia in sé ("è buona o cattiva"). Valuta la SORPRESA
rispetto a ciò che il mercato si aspettava e il POTENZIALE IMPATTO sui prezzi.

Ragiona in questo ordine:
1. CHE COSA è successo (fatto osservabile).
2. COSA si aspettava il mercato: cerca nel testo riferimenti espliciti alle attese
   ("beats/misses expectations", "above/below consensus", "guidance raised/cut").
   Se il testo NON fornisce il confronto con le attese, la sorpresa è debole:
   abbassa la confidence e mantieni l'impatto moderato.
3. SORPRESA: quanto l'esito si discosta dalle attese (molto positiva / positiva /
   neutrale / negativa / molto negativa).
4. MECCANISMO: perché questo dovrebbe muovere i prezzi (catena causale,
   es. inflazione ↑ → tassi attesi ↑ → costo del capitale ↑ → azioni growth ↓).
5. CHI GUADAGNA E CHI PERDE: individua le aree geografiche colpite. Per notizie
   specifiche dell'azienda usa principalmente l'area {area}; per notizie macro
   (Fed, inflazione, tassi, petrolio) indica tutte le aree colpite.

Restituisci UN UNICO oggetto JSON valido (senza blocchi markdown di contorno) con esattamente questa struttura:
{
  "impact_score": <float da -5.0 a +5.0; il segno indica la direzione, la magnitudo la forza della sorpresa>,
  "impact_duration": "brief" per effetto di poche ore, "medium" per giorni, "long" per settimane/mesi,
  "impact_surface": "<CSV di aree colpite tra: usa, eu, asia, emerging, other>",
  "confidence": <float tra 0.0 e 1.0>,
  "catalyst": "<breve etichetta del catalizzatore, es. 'Earnings beat', 'Regulatory risk', 'Macro data', 'Product launch' o 'General'>",
  "rationale_it": "<2-3 frasi professionali in lingua italiana: la sorpresa rispetto alle attese, il meccanismo economico e quale tesi di investimento supporta>"
}
```

### 2.2 Salvataggio in MariaDB
L'output JSON restituito da DeepSeek viene validato (clamp di `impact_score` a
±5.0, durata in `{brief, medium, long}`, superficie filtrata sui codici area) e
salvato nella tabella `news_sentiment`, legata alla notizia in `news`.

---

## 3. Generazione del Segnale Ibrido (`portfolio_signal`)

Una volta analizzate le notizie del giorno per un dato strumento:

1. **Candidati**: si raccolgono le notizie degli ultimi 14 giorni (finestra di
   retention) che (a) sono taggate direttamente sullo strumento, oppure (b) hanno
   un `impact_surface` che copre l'`area` dello strumento (**propagazione macro**).
2. **Gate di confidenza**: le notizie con `confidence < 0.30` vengono scartate
   (contributo nullo).
3. **Decadimento temporale** (`impact_decay`): ogni notizia contribuisce in modo
   proporzionale a quanto è ancora "fresca" rispetto alla sua durata attesa:

   ```text
   decay = max(0, 1 - età_notizia_in_ore / orizzonte)
   orizzonte: brief=24h · medium=96h · long=336h
   ```

4. **Contributo per notizia**:

   ```text
   contributo = (impact_score / 5.0) × 0.20 × confidence × decay × fattore_direzione
   fattore_direzione = 1.0 (diretta) · 0.5 (propagata via impact_surface)
   ```

5. **Modificatore**: `llm_sentiment_modifier = clamp(Σ contributi, ±0.20)`.
6. **Segnale finale**: `final_signal = quant_score + llm_sentiment_modifier`.
7. **Breakdown**: per ogni notizia che ha contribuito si salva un record JSON in
   `portfolio_signal.sentiment_breakdown` (titolo, `impact_score`, durata,
   confidenza, età in ore, `decay`, contributo) per il dettaglio nella webapp.
8. L'attributo `ai_rationale` viene consolidato unendo i titoli delle notizie che
   hanno contribuito con il loro `impact_score` e `decay`.

---

## 4. Generazione Riassunti Markdown (`job summaries`)

Parallelamente all'analisi quantitativa, il batch genera un documento **Markdown giornaliero** per ciascun portafoglio, pensato per la lettura rapida dell'utente.

### Esempio di Markdown Generato (`news_summary`):
```markdown
# Riepilogo Notizie & Sentiment — Azionario Globale — 2026-08-08

### AAPL — Apple Inc.
- **Apple announces breakthrough in custom silicon AI accelerators** 🟢 *BULLISH +3.8 · durata media (82%)*
  - *Editore:* Reuters · *Data:* 2026-08-08 06:30
  - *Analisi IA:* L'annuncio supera le attese del mercato sul segmento AI edge, rafforzando il vantaggio competitivo e sostenendo una revisione al rialzo delle stime di margine.
  - [Leggi originale](https://finance.yahoo.com/news/...)

### QQQ — Invesco QQQ Trust
- **Tech sector awaits upcoming inflation prints** 🟡 *NEUTRAL +0.4 · durata breve (61%)*
  - *Editore:* Bloomberg · *Data:* 2026-08-08 04:15
  - *Analisi IA:* Clima di attesa sui mercati azionari tecnologici in vista dei dati macroeconomici statunitensi; assenza di driver direzionali univoci.
  - [Leggi originale](https://finance.yahoo.com/news/...)
```

Questo file viene salvato sia nella colonna `markdown` della tabella `news_summary` sia esportato in una cartella locale per eventuale archivio.

---

## 5. Retention e Pulizia

Le notizie (e le relative `news_sentiment` in cascata) vengono conservate per **14 giorni**
per default. Il job batch `cleanup` (vedi `07-operativita-batch.md`) elimina le notizie
più vecchie del periodo di retention e rimuove anche i file cache parquet scaduti in `tmp/`
(`*_news.parquet` e `*_gnews.parquet`). Il periodo è configurabile con `--days`.
