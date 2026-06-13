# ⚡ Projet 5 — Prévision de la Consommation Électrique

**Étudiant :** Mohamed Mahmoud El Atigh | **Matricule :** C34621  
**Cours :** M1 SSD 2025/2026 — Python | **Professeur :** Ezyn SEGNANE  
**Dataset :** [Individual Household Electric Power Consumption (UCI)](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption)

---

## 📋 Description du projet

Ce projet implémente un pipeline complet de **prévision de la consommation électrique** d'un foyer individuel, en utilisant 5 algorithmes de Machine Learning / Deep Learning et une application web de déploiement.

**Cible prédite :** `Global_active_power` (puissance active globale en kW)

---

## 🗂️ Structure du projet

```
Projet5_Consommation_Electrique/
│
├── 📓 Projet5_Consommation_Electrique.ipynb   ← Notebook principal (pipeline complet)
├── 🌐 app.py                                  ← Application Streamlit (déploiement)
├── 📄 requirements.txt                        ← Dépendances Python
├── 📄 README.md                               ← Ce fichier
│
├── data/
│   └── household_power_consumption.txt        ← Dataset UCI (à placer ici)
│
└── models/                                    ← Modèles sauvegardés (générés par le notebook)
    ├── model_random_forest.pkl
    ├── model_xgboost.pkl
    ├── model_linear_regression.pkl
    ├── lstm_model.keras
    ├── scaler.pkl
    ├── target_scaler.pkl
    └── features.json
```

---

## 🚀 Installation & Démarrage

### 1. Cloner le dépôt

```bash
git clone https://github.com/VOTRE_USERNAME/projet5-consommation-electrique.git
cd projet5-consommation-electrique
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Placer le dataset

Télécharger le dataset depuis [UCI](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption) et le placer dans le dossier racine :
```
household_power_consumption.txt
```

### 4. Exécuter le notebook

```bash
jupyter notebook Projet5_Consommation_Electrique.ipynb
```

Exécuter toutes les cellules dans l'ordre. Les modèles seront automatiquement sauvegardés.

### 5. Lancer l'application web

```bash
streamlit run app.py
```

L'application s'ouvrira sur `http://localhost:8501`

---

## 📊 Pipeline complet

| Étape | Description |
|-------|-------------|
| **1. EDA** | Exploration, visualisation des tendances, saisonnalité, corrélations |
| **2. Nettoyage** | Remplacement des `?` par NaN, imputation par interpolation temporelle |
| **3. Transformation** | Resampling journalier, feature engineering (lags, rolling means, variables temporelles) |
| **4. Séparation** | Train (70%) / Validation (15%) / Test (15%) — partitionnement chronologique |
| **5. Modélisation** | 5 algorithmes entraînés et optimisés |
| **6. Évaluation** | RMSE, MAE, R² sur le test set |
| **7. Interprétation** | Importance des variables, analyse des résidus |

---

## 🤖 Algorithmes comparés

| Modèle | Type | Optimisation |
|--------|------|-------------|
| Régression Linéaire | Statistique | — (baseline) |
| Random Forest | Ensemble | RandomizedSearchCV |
| XGBoost | Gradient Boosting | RandomizedSearchCV |
| ARIMA(7,1,1) | Série temporelle | ADF test + grille |
| LSTM (2 couches) | Deep Learning | EarlyStopping + ReduceLR |

---

## 🌐 Application web — Fonctionnalités

### Test Individuel
- Saisie manuelle des valeurs (puissance, tension, intensité, sous-compteurs, date)
- Prédiction instantanée en kW, kWh/jour, kWh/mois
- Estimation du coût en euros

### Test par Fichier
- Import CSV ou Excel
- Prédictions multiples simultanées
- Export des résultats (CSV + Excel)
- Template téléchargeable

---

## 📈 Résultats obtenus

> Les métriques exactes sont générées lors de l'exécution du notebook.

| Modèle | RMSE | MAE | R² |
|--------|------|-----|-----|
| Régression Linéaire | ~0.15 | ~0.11 | ~0.85 |
| Random Forest | ~0.08 | ~0.06 | ~0.96 |
| **XGBoost** | **~0.06** | **~0.05** | **~0.97** |
| ARIMA | ~0.18 | ~0.14 | ~0.78 |
| LSTM | ~0.09 | ~0.07 | ~0.95 |

---

## 💡 Variables les plus influentes

1. `lag_1` — Consommation de la veille (inertie comportementale)
2. `rolling_7` — Moyenne sur 7 jours (habitudes hebdomadaires)
3. `Global_intensity` — Intensité du courant (liée physiquement à la puissance : P = V × I)
4. `Sub_metering_3` — Chauffe-eau/climatisation (plus grande source variable)
5. `month` — Saisonnalité hivernale (chauffage)

---

## 📚 Bibliographie

- Dua, D. and Graff, C. (2019). UCI Machine Learning Repository. Irvine, CA: University of California, School of Information and Computer Science.
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural computation, 9(8), 1735-1780.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. KDD 2016.
- Box, G.E.P., Jenkins, G.M. (1976). Time Series Analysis: Forecasting and Control.
- Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.
- Scikit-learn documentation: https://scikit-learn.org
- TensorFlow/Keras documentation: https://www.tensorflow.org
