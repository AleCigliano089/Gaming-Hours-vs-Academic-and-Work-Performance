
# Gaming Hours vs. Academic & Work Performance

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Stato](https://img.shields.io/badge/Status-Completato-brightgreen)

## 📋 Panoramica del Progetto
Questo progetto analizza l'impatto delle abitudini di gioco sulle prestazioni accademiche e lavorative di studenti e professionisti. Utilizzando un approccio di machine learning, lo studio mira a predire l'**Impatto sulle Prestazioni** (Negativo, Neutro, Positivo) basandosi su caratteristiche comportamentali come ore di gioco, qualità del sonno e livelli di stress.

Il progetto è stato realizzato presso l'**Università degli Studi di Salerno**.

### 🎯 Obiettivi
1.  **Classificazione:** Predire la categoria di `Performance_Impact` utilizzando un **Random Forest Classifier**.
2.  **Regressione:** Tentare di predire il punteggio esatto `Academic_or_Work_Score` utilizzando un **Random Forest Regressor**.
3.  **Analisi Dati:** Investigare la validità del dataset sintetico e i suoi pattern di correlazione.

---

## 📊 Il Dataset
**Fonte:** [Gaming Hours vs Academic & Work Performance (Kaggle)](https://www.kaggle.com/datasets/prince7489/gaming-hours-vs-academic-and-work-performance).

Il dataset contiene **1.000 righe** e **14 feature**. Si tratta di un **dataset sintetico**, generato considerando pattern identificati in studi scientifici piuttosto che recuperati direttamente tramite questionari reali.

### Feature Principali
* **Predittori:** `Daily_Gaming_Hours`, `Sleep_Hours`, `Age`, `Stress_Level`, `Focus_Level`.
* **Target (Classificazione):** `Performance_Impact` (Negative, Neutral, Positive).
* **Target (Regressione):** `Academic_or_Work_Score`.

### Pulizia dei Dati e Pre-processing
* **Valori Mancanti:** Nessuno trovato (Integrità completa verificata).
* **Outliers:** Nessun outlier rilevato tramite il metodo IQR.
* **Rimozione Feature:**
    * `User_ID`: Irrilevante per il modello.
    * `Weekly_Gaming_Hours`: Rimossa per perfetta correlazione (ridondanza) con le ore giornaliere.
    * `Productivity_Level`: Rimossa per alta multicollinearità (0.96) con il punteggio accademico.
* **Scaling:** Utilizzato MinMaxScaler per normalizzare le feature numeriche nel range [0, 1].

---

## ⚙️ Metodologia

### 1. Gestione del Bilanciamento delle Classi (SMOTENC)
Il dataset presentava un evidente sbilanciamento, con la classe **"Neutral"** che rappresentava il **76,2%** dei dati.

* **Strategia:** È stato scartato l'undersampling per evitare la perdita di informazioni.
* **Implementazione:** È stata utilizzata la tecnica **SMOTENC** (Synthetic Minority Over-sampling Technique Nominal Continous) applicata **esclusivamente al Training Set** per generare istanze sintetiche delle classi minoritarie ("Positive" e "Negative").
* **Risultato:** Uno spazio delle feature arricchito senza influenzare la fase di valutazione (Test Set).

### 2. Scelta del Modello
* **Classificatore:** `RandomForestClassifier` (Ottimizzato tramite GridSearchCV).
* **Regressore:** `RandomForestRegressor`.

---

## 📈 Risultati e Analisi

### 🏆 Modello di Classificazione (Random Forest)
Il modello di classificazione ha ottenuto risultati quasi perfetti (**99-100%** su Accuracy, Precision e Recall).

#### Perché un'accuratezza così alta?
L'analisi dei confini decisionali ha rivelato che i dati sintetici seguono regole rigide e nette:
* **Positive:** Soggetti che giocano ≤ 2 ore e dormono > 7 ore.
* **Negative:** Soggetti che giocano ≥ 4 ore e dormono < 6 ore.
* **Neutral:** Tutte le restanti combinazioni.

Poiché queste classi sono linearmente separabili nella logica di generazione sintetica, il modello è stato in grado di classificarle perfettamente.

### 📉 Modello di Regressione (Random Forest)
Il modello di regressione ha tentato di predire il valore di `Academic_or_Work_Score`.
* **MAE (Errore Medio Assoluto):** ~10.05.
* **R² Score:** ~0.0022.

**Conclusioni:** Il modello di regressione ha avuto performance scarse, paragonabili a una scelta casuale. Questo indica che, sebbene il dataset abbia regole rigide per le *categorie* (Impatto), manca della correlazione causale granulare necessaria per predire specifici *punteggi* numerici.

---

## 🛠️ Installazione e Utilizzo

### Prerequisiti
* Python 3.x
* pandas 
* numpy 
* seaborn & matplotlib
* scikit-learn 
* imbalanced-learn 

### Installazione
```bash
pip install pandas numpy seaborn matplotlib scikit-learn imbalanced-learn
```

### Esecuzione
```bash
python analysis_script.py
```




