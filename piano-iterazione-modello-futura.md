# Piano Operativo: Nuova Iterazione del Modello Quant

Questo documento definisce il piano per la prossima iterazione del modello quant di HAYAI, dopo la validazione cronologica che ha mostrato che il modello attuale non è ancora pronto per la produzione. L’obiettivo è testare una nuova definizione del target e un nuovo set di feature che migliori la potenza predittiva sul holdout senza introdurre leakage o contaminazione da split casuale.

---

## Task 1: Ridefinire il target per un segnale più robusto
- **Stato**: todo
- **Scopo**: riconsiderare il target del modello per allinearlo meglio alla vera capacità decisionale del portafoglio e ridurre il rumore legato a rendimenti logaritmici troppo sensibili a brevi oscillazioni.
- **Risultato atteso**: una scelta documentata di target (es. ritorno forward 5/10/20 giorni, media mobile o score rankizzato), con spiegazione della relazione tra target e decisione long/short e con evidenza che il target non implichi look-ahead o leakage.
- **Test**: confronto tra diverse definizioni di target su dataset cronologico con metriche di Spearman, R² e hit rate; scelta del target con performance migliore in holdout e minore variabilità.
- **Todolist**:
  - [ ] analizzare il target attuale `clip(log(close(t+5)/close(t)) / vol_20, -3, 3)` e identificare i punti deboli rispetto al ranking long/short.
  - [ ] definire 2-3 alternative di target (es. ritorno forward 5gg normalizzato, momentum multi-periodo, target binario rankizzato).
  - [ ] validare i target alternativi sullo stesso holdout cronologico usando il workflow di diagnostic.
  - [ ] scegliere il target più stabile e meno rumoroso per il ciclo successivo.

## Task 2: Costruire un nuovo feature set con maggiore capacità di separazione
- **Stato**: todo
- **Scopo**: introdurre feature che catturino meglio struttura di mercato, trend, volatilità e qualità del ranking senza appesantire il modello o introdurre leakage.
- **Risultato atteso**: un feature set rivisto, documentato e compatibile con il pipeline cronologico, con feature calcolate solo usando dati disponibili al tempo `t`.
- **Test**: confronto del feature set nuovo vs attuale su holdout cronologico e analisi di importanza/varianza per i principali segnali.
- **Todolist**:
  - [ ] raccogliere le feature attive e identificare le più deboli o ridondanti.
  - [ ] aggiungere feature di trend a più scale, regime di mercato e qualità del momentum (es. z-score multi-range, skewness locale, qualità del breakout, resistenza/support).
  - [ ] rimuovere feature senza potere discriminante o con forte collinearità.
  - [ ] mantenere il feature set compatibile con la logica `fit-on-train` e il pannello completo per data.

## Task 3: Separare correttamente fit, validazione e test su window temporali reali
- **Stato**: todo
- **Scopo**: garantire che il nuovo ciclo di training usi un protocollo di validazione cronologica corretto, senza contaminazione tra set di fit, validazione e test.
- **Risultato atteso**: una pipeline valida per ogni run, con cutoffs chiari e log del protocollo di split utilizzato.
- **Test**: verifica di split cronologico e controllo del contenuto dei set train/val/test con grep/log e asserts di dimensione e date.
- **Todolist**:
  - [ ] definire le finestre train / validation / test in base ai dati storici disponibili.
  - [ ] salvare in `config.json` le date di cut e la metodologia del preprocessing per ogni artefatto.
  - [ ] verificare che winsorization e scaling siano calcolati solo sul train e non sul test.
  - [ ] documentare il protocollo per ancorare la decisione finale al holdout più recente.

## Task 4: Implementare il nuovo job di training e comparazione
- **Stato**: todo
- **Scopo**: costruire un nuovo job di training dedicato al nuovo ciclo e un report comparativo tra la baseline attuale e il nuovo modello per scegliere se procedere o tornare indietro.
- **Risultato atteso**: un artefatto numerato e un report con metriche coerenti tra train, validation e test; il nuovo modello deve avere un confronto su stesso dataset e stessa metodologia.
- **Test**: esecuzione del job e generazione del report con comparazione `baseline vs candidate`, verificando che i risultati affrontino lo stesso holdout.
- **Todolist**:
  - [ ] creare un nuovo job `train_candidate_<nn>` o estensione del job clean con nuova target/feature set.
  - [ ] serializzare artefatto e config in una directory dedicata.
  - [ ] eseguire diagnostic comparativo con `v4`, `v5_clean_time` e nuovo candidato.
  - [ ] valutare se il nuovo modello migliora in modo reale sul holdout cronologico.

## Task 5: Testare il ruolo delle notizie e del giudizio LLM in mixed signal
- **Stato**: todo
- **Scopo**: verificare se il modello quant è realmente debole o se la copertura delle notizie e la qualità del sentiment LLM stanno limitando il potenziale del segnale ibrido.
- **Risultato atteso**: una stima del contributo netto del modello quant rispetto al modificatore LLM, con evidenza su copertura, qualità e beneficio netto.
- **Test**: confronto tra quant-only, llm-only e hybrid su stesso periodo, con analisi di cobertura delle news e correnti di sentiment.
- **Todolist**:
  - [ ] verificare la copertura notizie storiche per i tickers del periodo di test.
  - [ ] confrontare i segnali puri quant con quelli hybrid e lo stesso target cronologico.
  - [ ] misurare se l’LLM aggiunge valore reale o se il modello quant non è abbastanza solido da essere usato in modo affidabile.
  - [ ] riportare il risultato in un mini-report di triage del modello.

## Task 6: Decisione di gate finale e prossimi step
- **Stato**: todo
- **Scopo**: stabilire un gate chiaro per decidere se la nuova iterazione è pronta per un ulteriore ciclo o se si deve tornare alla fase di feature design.
- **Risultato atteso**: una decisione documentata di `accetta`, `rigetta` o `ripeti con nuova ipotesi`, basata su evidenze quantitative e non su impressione.
- **Test**: confronto finale su metriche come Spearman, R², hit-rate, drawdown e stabilità in finestre temporali separate.
- **Todolist**:
  - [ ] definire criteri minimi di gate (es. Spearman positivo stabile, R² migliorato, hit rate superiore alla baseline di settore, nessun leak, news coverage sufficiente).
  - [ ] decidere se il candidato supera o no il gate.
  - [ ] preparare la lista di azioni per la prossima iterazione in base alla decisione.

---

## Checklist di avanzamento
- [ ] Task 1 completata
- [ ] Task 2 completata
- [ ] Task 3 completata
- [ ] Task 4 completata
- [ ] Task 5 completata
- [ ] Task 6 completata

Questo piano è il riferimento operativo per l’iterazione successiva del modello quant e per la decisione finale del nuovo candidato production-ready.
