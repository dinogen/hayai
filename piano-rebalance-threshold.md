# Piano Operativo: Soglia di Tolleranza Riallineamento Portafoglio (`piano-rebalance-threshold.md`)

Questo documento definisce il piano per introdurre una soglia configurabile in euro (`rebalance_threshold_eur`) per ignorare variazioni minori (micro-aggiustamenti) nella tabella di riconciliazione e nelle raccomandazioni, gestibile dalla pagina di configurazione.

---

## Task 1: Aggiornamento Schema Database & Backend (Colonna `rebalance_threshold_eur`)
- **Stato**: done
- **Scopo**: Aggiungere il campo `rebalance_threshold_eur` (default 50.00) alla tabella `portfolio` nel database MariaDB e aggiornare i router API di configurazione (`config.py`) e raccomandazioni (`portfolios.py`).
- **Risultato atteso**: La tabella `portfolio` supporta la soglia in euro; l'API di configurazione permette di leggerla e modificarla; l'endpoint di raccomandazioni utilizza la soglia per marciare come `hold` (`mantieni (invariato)`) le variazioni inferiori alla soglia.
- **Todolist**:
  - [x] Eseguire alter table in MariaDB (o query di aggiornamento) per aggiungere `rebalance_threshold_eur DECIMAL(10,2) NOT NULL DEFAULT 50.00` alla tabella `portfolio`.
  - [x] Aggiornare `api/routers/config.py` per includere `rebalance_threshold_eur` nelle richieste di lettura e salvataggio configurazione.
  - [x] Aggiornare `api/routers/portfolios.py` (`get_latest_recommendations`) per leggere `rebalance_threshold_eur` dal portafoglio e applicare la regola di tolleranza sulle variazioni (`diff_eur < threshold` -> `hold`).

## Task 2: Aggiornamento Frontend Angular (`ConfigComponent` & `RecommendationsComponent`)
- **Stato**: done
- **Scopo**: Aggiungere il campo di input numerico per la soglia di tolleranza in euro nella pagina di configurazione (`ConfigComponent`) e aggiornare la visualizzazione in `RecommendationsComponent`.
- **Risultato atteso**: L'utente può visualizzare e modificare la soglia di tolleranza da Configurazione; la tabella di riconciliazione riflette correttamente le variazioni filtrate dalla soglia.
- **Todolist**:
  - [x] Aggiornare `ConfigComponent` (`config.component.ts`) aggiungendo il campo di input per la soglia di tolleranza in euro e l'handler di salvataggio.
  - [x] Aggiornare `ApiService` (`api.service.ts`) se necessario per supportare l'aggiornamento della soglia.
  - [x] Verificare il corretto rendering delle azioni (`buy`, `sell`, `hold`) nella tabella di riconciliazione.

## Task 3: Build e Verifica Finale
- **Stato**: done
- **Scopo**: Compilare la webapp Angular e verificare il funzionamento end-to-end.
- **Risultato atteso**: Build Angular completata con successo (`npm run build`).
- **Todolist**:
  - [x] Eseguire `npm run build` in `hayai-new\web`.
  - [x] Verificare che non vi siano errori di compilazione TypeScript o template.
