# Détection de Transactions Atypiques dans les Données de Mobile Money

**Projet Intégrateur (Capstone) — Programme FRST**
Certificat en Sciences des Données, IA & Mathématiques pour l'IA
Faculté des Sciences, Université d'État d'Haïti (FDS-UEH)

---

## Équipe

| Nom | Rôle |
|---|---|
| JEAN-LOUIS Berckson Johnsly | Co-auteur |
| MORISSET Nherlyse | Co-auteur |
| SERVILUS Bendy | Co-auteur |

**Encadrement :** Programme FRST (partenariat FDS-UEH / BRH)
**Date de soumission du résumé :** 29 avril 2026
**Date limite de dépôt du projet final :** 17 juillet 2026, 23h50

---

## 1. Contexte et problématique

En Haïti, le mobile money (MonCash, NatCash) est devenu un pilier de l'inclusion financière, traitant quotidiennement des millions de transactions. Les plaintes récurrentes d'usagers concernant des opérations suspectes exposent les opérateurs à des risques financiers, réputationnels et de conformité (lutte contre le blanchiment, exigences BRH).

**Objectif du projet :** construire un pipeline capable d'identifier automatiquement les transactions atypiques (potentiellement frauduleuses), en combinant une approche supervisée (avec labels) et des méthodes non supervisées de détection d'anomalies, puis d'en évaluer et interpréter les résultats de façon critique.

---

## 2. Jeu de données

Faute d'accès à des données réelles labellisées de MonCash/NatCash (confidentialité), le projet s'appuie sur **PaySim**, un simulateur de transactions mobile money publié sur Kaggle, reconnu dans la littérature académique comme un proxy réaliste de ce type de service.

- **Source :** [PaySim — Synthetic Financial Datasets For Fraud Detection](https://www.kaggle.com/datasets/ealaxi/paysim1)
- **Fichier :** `PS_20174392719_1491204439457_log.csv`
- **Volume :** ~6,3 millions de transactions, 744 pas de temps (simulation de 30 jours)
- **Variable cible :** `isFraud` (0,13 % de fraude — classes très déséquilibrées)
- **Variables principales :** `step`, `type`, `amount`, `nameOrig`, `oldbalanceOrg`, `newbalanceOrig`, `nameDest`, `oldbalanceDest`, `newbalanceDest`, `isFraud`, `isFlaggedFraud`
- **Particularité du simulateur :** les fraudes n'apparaissent que sur les transactions de type `TRANSFER` et `CASH_OUT` (comportement documenté et vérifié empiriquement dans le notebook)

> ⚠️ Le CSV n'est pas versionné dans ce dépôt (fichier volumineux). À télécharger depuis Kaggle et placer dans `datasets/PS_20174392719_1491204439457_log.csv`.

---

## 3. Structure du dépôt

```
├── transaction-fraud-detection-capstone.ipynb   # Notebook principal (analyse complète)
├── datasets/
│   └── PS_20174392719_1491204439457_log.csv     # Dataset PaySim (à télécharger, non fourni)
├── cadrage_projet_capstone.docx                 # Note de cadrage FRST (consignes officielles)
├── RESUME_CapStone.pdf                          # Résumé du projet intégrateur (soumis le 29/04/2026)
├── FRST_SDIAM_Projets_Captone-Sujets.pdf        # Liste des sujets proposés par le programme
└── README.md                                    # Ce fichier
```

---

## 4. Contenu du notebook

Le notebook `transaction-fraud-detection-capstone.ipynb` est structuré en 11 sections :

1. **Chargement et qualité des données** — dimensions, types, valeurs manquantes, déséquilibre des classes
2. **Analyse exploratoire multivariée** — matrice de corrélation, boxplots montant × type × fraude, pairplot
3. **Feature engineering** — `errorBalanceOrig`, `errorBalanceDest`, `is_merchant`, `is_large_amount`
4. **Séparation Train / Test** — split stratifié 70/30
5. **Pipeline de prétraitement** — `ColumnTransformer` (StandardScaler + OneHotEncoder)
6. **Modélisation supervisée** — Logistic Regression, Decision Tree, Random Forest (avec validation croisée)
7. **Évaluation approfondie** — courbes ROC, courbes Précision-Rappel, AUC (métriques adaptées au déséquilibre)
8. **Importance des variables** — classement des features les plus discriminantes (Random Forest)
9. **Analyse en Composantes Principales (ACP)** — variance expliquée, scree plot, projection 2D, cercle des corrélations
10. **Détection d'anomalies non supervisée** — Isolation Forest et DBSCAN (sans utiliser le label)
11. **Synthèse, limites et considérations éthiques** — comparaison des approches, biais potentiels, usage responsable

---

## 5. Méthodologie

Le projet combine deux logiques complémentaires :

| Approche | Algorithmes | Usage du label | Objectif |
|---|---|---|---|
| Supervisée | Decision Tree, Random Forest, Logistic Regression | Oui | Maximiser la précision de détection sur les schémas de fraude connus |
| Non supervisée | Isolation Forest, DBSCAN, ACP | Non | Détecter des anomalies sans historique labellisé (utile face à de nouveaux schémas de fraude) |

> **Note sur le pivot méthodologique :** le résumé initial (avril 2026) prévoyait une approche exclusivement non supervisée sur données synthétiques génériques. L'utilisation du dataset PaySim, réaliste et labellisé, a permis d'enrichir le travail avec une approche supervisée robuste tout en conservant ACP, Isolation Forest et DBSCAN annoncés dans les mots-clés du sujet. Cette évolution est justifiée en détail dans la section 11 du notebook.

---

## 6. Prérequis techniques

```bash
pip install pandas numpy scikit-learn seaborn matplotlib plotly
```

- Python ≥ 3.9
- Jupyter Notebook ou JupyterLab
- ~4 Go de RAM disponibles recommandés (dataset de 6,3M lignes)
- Temps d'exécution complet : plusieurs minutes (Random Forest et Isolation Forest sur l'ensemble du jeu de données)

---

## 7. Résultats clés (résumé)

- Les fraudes se concentrent exclusivement sur `TRANSFER` et `CASH_OUT`, avec des montants systématiquement plus élevés que les transactions légitimes.
- Le schéma de fraude dominant correspond à un **vidage de compte** (`amount ≈ oldbalanceOrg`), détecté via l'analyse multivariée et confirmé par la variable `errorBalanceOrig`.
- Les variables de solde brutes sont fortement multicolinéaires (`oldbalanceOrg`/`newbalanceOrig` : r=1.00 ; `oldbalanceDest`/`newbalanceDest` : r=0.98) — justifiant le recours à l'ACP.
- La fraude n'est pas linéairement séparable à partir des variables brutes (corrélations proches de 0 avec `isFraud`), justifiant l'usage de modèles non linéaires.
- Le mécanisme `isFlaggedFraud` déjà intégré au système ne capte presque aucune fraude réelle (r=0.04 avec `isFraud`) — la limite que ce projet cherche justement à combler.

*(Chiffres précis de performance — precision, rappel, F1, AUC — à consulter dans la section 7 du notebook après exécution.)*

---

## 8. Limites

- Déséquilibre extrême des classes (0,13 % de fraude) : accuracy seule non pertinente.
- Données synthétiques : réalistes mais non spécifiques au contexte haïtien (agents, zones géographiques locales).
- Certaines variables engineerées peuvent créer une fuite de données par construction du simulateur ; leur poids doit être interprété avec prudence.
- DBSCAN et Isolation Forest appliqués sur échantillon/projection réduite pour des raisons de scalabilité.

---

## 9. Considérations éthiques

Un score de risque produit par ce type de modèle doit rester une **aide à la décision** pour les équipes conformité, et non une décision automatique de blocage, afin d'éviter de pénaliser des usagers légitimes (faux positifs) et de préserver l'accès aux services financiers.

---

## 10. Livrables du projet

- [x] Notebook d'analyse (`.ipynb`)
- [ ] Rapport écrit structuré (sections 4.1 à 4.8 selon la note de cadrage)
- [ ] Présentation PowerPoint (soutenance)
- [ ] Vidéo de présentation
- [ ] Dépôt final : 1 PDF (mémoire + annexes) + 1 PPTX + 1 vidéo, avant le **17 juillet 2026, 23h50**

---

## Références

- Lopez-Rojas, E. A., Elmir, A., & Axelsson, S. (2016). *PaySim: A financial mobile money simulator for fraud detection.* The 28th European Modeling and Simulation Symposium (EMSS), Larnaca, Chypre.
- Dataset : [https://www.kaggle.com/datasets/ealaxi/paysim1](https://www.kaggle.com/datasets/ealaxi/paysim1)
