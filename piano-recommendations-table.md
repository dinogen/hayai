# Piano Operativo: Tabella Full Outer Join (Portafoglio vs Raccomandazioni) in Recommendations (`piano-recommendations-table.md`)

Questo documento definisce il piano per aggiungere in fondo alla pagina delle raccomandazioni una tabella di Full Outer Join tra il portafoglio attuale e le raccomandazioni, con azioni e messaggi operativi chiari (compra, vendi, chiudi).

---

## Task 1: Aggiornamento Backend API (Full Outer Join Posizioni & Raccomandazioni)
- **Stato**: done
- **Scopo**: Modificare l'endpoint `/portfolios/{code}/recommendations/latest` in `portfolios.py` per unire le posizioni correnti (`portfolio_position`) e le raccomandazioni (`portfolio_recommendation`), calcolando la differenza, l'azione (`buy`, `sell`, `close`, `hold`) e il messaggio formattato ("compra tot", "vendi tot", "chiudi questa posizione").
- **Risultato atteso**: L'API restituisce un campo aggiuntivo `reconciliation` contenente la lista degli asset con ticker, quote possedute, quote raccomandate, differenza e messaggio operativo.
- **Todolist**:
  - [x] Interrogare le posizioni correnti del portafoglio nell'ultima data disponibile.
  - [x] Interrogare le raccomandazioni dell'ultima data.
  - [x] Eseguire il merge full outer join in Python per combinare strumenti posseduti e raccomandati.
  - [x] Calcolare delta, azione (`buy`/`sell`/`close`/`hold`) e stringa del messaggio.
  - [x] Includere `reconciliation` nel JSON di risposta dell'endpoint.

## Task 2: Aggiornamento Frontend Angular (`RecommendationsComponent`)
- **Stato**: done
- **Scopo**: Aggiungere in fondo alla pagina delle raccomandazioni una tabella HTML pulita e responsiva (`overflow-x-auto`) che mostra la tabella di riconciliazione/riallineamento con colonne: Ticker/Strumento, Quote Possedute, Quote Raccomandate, Azione/Differenza (con badge colorati).
- **Risultato atteso**: Tabella visibile in fondo alla pagina con messaggi chiari ("chiudi questa posizione", "compra X", "vendi X").
- **Todolist**:
  - [x] Aggiornare il template HTML di `RecommendationsComponent` inserendo la sezione della tabella di riconciliazione.
  - [x] Applicare stili CSS in linea con il Design System (badge verde per buy, rosso per sell/chiudi, grigio per hold).
  - [x] Gestire i dati provenienti dal nuovo campo API `reconciliation`.

## Task 3: Build e Verifica Finale
- **Stato**: done
- **Scopo**: Verificare la corretta compilazione della webapp Angular e il funzionamento dell'API FastAPI.
- **Risultato atteso**: Build completata con successo senza errori di compilazione.
- **Todolist**:
  - [x] Eseguire `npm run build` in `hayai-new\web`.
  - [x] Verificare l'assenza di errori TypeScript o template.
