# 📊 Prédiction des Résultats Électoraux – Big Data & Machine Learning

> MSPR Bloc 3 – Piloter l'informatique décisionnelle d'un S.I (Big Data & Business Intelligence)  
> Certification Expert Informatique SI – RNCP Niveau 7 | EPSI

---

## 📋 Contexte du projet

Dans le cadre de la certification RNCP Niveau 7, ce projet consiste à développer une 
**preuve de concept (POC)** de prédiction électorale par Machine Learning pour la startup 
fictive **Elexxion**, spécialisée dans le conseil en campagnes électorales.

L'objectif : démontrer la faisabilité d'un modèle prédictif basé sur des indicateurs 
socio-économiques (emploi, sécurité, population, vie économique...) corrélés aux résultats 
des élections passées, afin d'anticiper les tendances futures.

---

##  Objectifs du POC

- Sélectionner un secteur géographique cible pour la POC
- Identifier et collecter des jeux de données publics pertinents
- Établir des corrélations entre indicateurs socio-économiques et résultats électoraux
- Entraîner et comparer plusieurs modèles de Machine Learning supervisé
- Visualiser les résultats et proposer des prédictions à 1, 2 et 3 ans

---

##  Données utilisées

Sources officielles françaises (data.gouv.fr) :
- Résultats des élections passées par circonscription/département
- Données INSEE : population, emploi, vie économique
- Indicateurs de sécurité

---

##  Modèles de Machine Learning testés

| Modèle | Description |
|--------|-------------|
| **Random Forest** | Modèle d'ensemble par arbres de décision |
| **XGBoost** | Gradient Boosting optimisé |
| **Keras (Deep Learning)** | Réseau de neurones pour la prédiction |
| **KMeans** | Clustering des circonscriptions par profil socio-économique |

---

##  Résultats

- Comparaison des modèles sur plusieurs découpages (77%, 80/20, 2017-2022)
- Évaluation des performances via accuracy et matrices de corrélation
- Visualisations par parti politique (corrélations, heatmaps)
- Courbes d'apprentissage (loss/accuracy) pour le modèle Keras
- Clustering KMeans des profils électoraux

---

##  Stack technique

| Catégorie | Technologies |
|-----------|-------------|
| Langage | Python |
| Machine Learning | Scikit-learn, XGBoost, Keras/TensorFlow |
| Clustering | KMeans (Scikit-learn) |
| Data Processing | Pandas, NumPy |
| Visualisation | Matplotlib, Google LineChart |
| Versioning | GitHub |

---

## 📁 Structure du repo

```
mspr-bigdata-election-prediction/
├── README.md
├── docs/
│   └── rapport_final_mspr_bloc3.pdf
└── resultats/
├── correlations_par_parti/
├── KERAS/
├── KMEANS/
├── RANDOMFOREST/
└── XGBOOST/
```
---

## 👥 Réalisé par

Projet réalisé en équipe dans le cadre de la certification  
**Expert Informatique et Système d'Information – RNCP Niveau 7**  
EPSI | Promotion 2024-2026
