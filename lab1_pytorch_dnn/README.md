# Lab 1 — Deep Learning avec PyTorch : Régression & Classification multi-classes

## 1. Objectif

Se familiariser avec la bibliothèque **PyTorch** en réalisant deux tâches d'apprentissage supervisé
avec des architectures **DNN / MLP** :

| Partie | Tâche | Dataset | Cible |
|---|---|---|---|
| **1** | **Régression** | [NYSE Stock Prices](https://www.kaggle.com/datasets/dgawlik/nyse) | prix de clôture de J+1 |
| **2** | **Classification multi-classes** | [Machine Predictive Maintenance](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification) | type de panne (`Failure Type`) |

---

## 2. Contenu du lab

```text
lab1_pytorch_dnn/
├── Atelier1.pdf                                # énoncé du lab
├── partie1_regression_nyse.ipynb               # Partie 1 : régression
├── partie2_classification_maintenance.ipynb    # Partie 2 : classification multi-classes
└── README.md                                   # ce rapport
```

Chaque notebook suit **l'ordre exact des questions de l'énoncé** : il y a un tableau récapitulatif
en haut de chaque notebook indiquant quelle cellule répond à quelle question.

---

## 3. Exécution

Les notebooks sont conçus pour **Google Colab**.

1. Ouvrir [Google Colab](https://colab.research.google.com/) → *Fichier → Importer un notebook*,
   et charger le notebook souhaité.
2. **Télécharger les datasets** depuis Kaggle (bouton *Download* sur les pages ci-dessus) :
   - `prices.csv` → [NYSE](https://www.kaggle.com/datasets/dgawlik/nyse)
   - `predictive_maintenance.csv` → [Maintenance](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification)
3. **Déposer le CSV à la racine de `/content`** : panneau *Fichiers* (📁 à gauche) → bouton
   *Importer* (↥) → sélectionner le fichier.
4. Exécuter les cellules **dans l'ordre** (*Exécution → Exécuter tout*).

> La cellule de chargement affiche le nombre de lignes chargées : il doit indiquer **851 264** pour
> `prices.csv` et **10 000** pour `predictive_maintenance.csv`. Un fichier tronqué, par exemple après un
> upload interrompu, fausserait tous les résultats.

**Exécution locale :** installer les dépendances puis lancer Jupyter.

```bash
pip install -r ../requirements.txt
jupyter lab
```

Les notebooks utilisent le chemin `/content` (spécifique à Colab). En local, modifier la ligne
`chemin = Path("/content") / "..."` en `chemin = Path("data") / "..."` après avoir placé les CSV
dans un dossier `data/`.

---

## 4. Partie 1 — Régression (NYSE)

### Démarche

| Question | Traitement |
|---|---|
| 1. EDA | Un seul titre retenu (**AAPL**) → une seule série temporelle, simple à analyser. Étude du prix, du volume, des rendements, des corrélations. |
| 2. Architecture DNN | **Entrées** = les 5 mesures du jour J (`open`, `high`, `low`, `close`, `volume`). **Cible** = le `close` de J+1. 2 couches cachées, activation ReLU. |
| 3. GridSearch | 8 combinaisons : 2 architectures × 2 learning rates × 2 optimiseurs, validation croisée à 3 blocs. Sélection sur le **R²**. |
| 4. Courbes | **Loss (MSE) / Epochs** et **R² / Epochs**, pour le train et le test. |
| 5. Régularisation | 4 configurations comparées : sans, Dropout 0.3, L2, Dropout + L2. |

### Points méthodologiques

- **Découpage chronologique** 80 % / 20 % : on ne mélange jamais une série temporelle, sinon le
  modèle s'entraîne sur le futur pour prédire le passé.
- **Normalisation** des entrées *et* de la cible (sans cela, la loss vaut environ 10 000 et
  l'entraînement est instable). Le scaler est ajusté **sur le train uniquement**.
- En régression, **l'« accuracy » n'existe pas** : elle est remplacée par le **R²**.

### Résultats

Données : **AAPL**, 1 762 séances (2010-01-04 → 2016-12-30), découpage 80 % / 20 % → **1 408 train / 353 test**.

| Jeu | MSE | RMSE | MAE | R² |
|---|---|---|---|---|
| Train | 357.83 | 18.92 | 8.36 | 0.988 |
| Test | 3.88 | 1.97 | 1.51 | 0.939 |

**Meilleurs hyperparamètres (GridSearchCV, cv=3) :**

| Hyperparamètre | Valeur |
|---|---|
| Architecture (`couches`) | `(128, 64, 32)` |
| Learning rate (`lr`) | `0.005` |
| Optimiseur | `adam` |
| Epochs | `50` |

Meilleur R² en validation croisée : **0.9771**

### Interprétation

- La loss décroît puis se stabilise ; les courbes **train et test restent proches** (R² 0.988 contre
  0.939) → pas de surapprentissage marqué.
- Le **R² du test est très élevé (0.939)**. Explication : le prix de demain ressemble beaucoup à celui
  d'aujourd'hui (les prix sont **autocorrélés**). Le modèle reproduit donc surtout la **tendance** et non
  les mouvements quotidiens.
- Le **MAE = 1,51 $** est la métrique la plus parlante : le modèle se trompe en moyenne de **1,51 $**
  sur une action qui vaut environ 300 $, soit **moins de 0,5 % d'erreur**.
- Le **MSE du train (357) est plus grand que celui du test (3,9)** : ce n'est pas du surapprentissage
  mais un effet de période — les prix de 2010-2016 varient dans une amplitude bien plus grande que
  ceux de la période de test (2015-2016).

### Régularisation

| Configuration | RMSE train | R² train | RMSE test | R² test |
|---|---|---|---|---|
| Sans régularisation | 16.32 | 0.991 | 1.89 | 0.944 |
| Dropout 0.3 | 17.06 | 0.990 | 2.98 | 0.861 |
| **L2 (`weight_decay=1e-4`)** | 16.41 | 0.991 | **1.87** | **0.945** |
| Dropout + L2 | 16.88 | 0.990 | 3.25 | 0.834 |

**Interprétation :** le réseau **ne surapprend pas** — l'écart train/test est faible (0.991 contre
0.944). Or la régularisation sert justement à réduire cet écart : il n'y a donc presque rien à
corriger. C'est pourquoi :

- le **Dropout 0.3 dégrade nettement le test** (0.861) : il est **trop fort** pour un réseau de
  3 couches → **sous-apprentissage** ;
- le **L2 à 1e-4 est neutre** (0.944 → 0.945) : il faudrait un `weight_decay` plus fort (1e-3, 1e-2)
  pour observer un effet.

**Conclusion :** sur ce problème, la régularisation **n'apporte aucun gain**. Elle n'aide que lorsqu'il
existe un surapprentissage à corriger — et ici le réseau est déjà suffisamment simple
(5 entrées, 3 couches).

---

## 5. Partie 2 — Classification multi-classes (Maintenance prédictive)

### Démarche

| Question | Traitement |
|---|---|
| 1. Prétraitement | Suppression des identifiants `UDI` / `Product ID`, suppression des doublons, one-hot encoding de `Type`, **standardisation** (moyenne 0, écart-type 1) ajustée sur le train. |
| 2. EDA | Mise en évidence du **déséquilibre des classes** et des variables discriminantes (Torque, Rotational speed, Tool wear). |
| 3. Augmentation | **SMOTE** sur le train uniquement, pour équilibrer les classes. |
| 4. Architecture DNN | MLP à sortie multiple (6 logits) + **`CrossEntropyLoss`**. |
| 5. GridSearch | 8 combinaisons, sélection sur le **F1 macro**, validation croisée **stratifiée**. |
| 6. Courbes | **Loss / Epochs** et **Accuracy / Epochs**, train et test. |
| 7. Métriques | Accuracy, **sensibilité (recall)**, précision, **F1**, rapport par classe, matrice de confusion. |
| 8. Régularisation | Mêmes 4 configurations que la partie 1. |

### Résultats

Données : 10 000 pièces, **8 entrées** (5 capteurs + `Type` en one-hot), 6 classes.
Découpage stratifié 80 % / 20 % → **8 000 train / 2 000 test**, puis SMOTE sur le train seul
(46 332 exemples équilibrés, 7 722 par classe).

| Jeu | Accuracy | Sensibilité (recall) | Précision | F1 (macro) |
|---|---|---|---|---|
| Train (équilibré SMOTE) | 0.997 | 0.997 | 0.997 | 0.997 |
| **Test (réel)** | **0.965** | **0.594** | **0.549** | **0.568** |

**Meilleurs hyperparamètres (GridSearchCV, StratifiedKFold) :**

| Hyperparamètre | Valeur |
|---|---|
| Architecture (`couches`) | `(128, 64, 32)` |
| Learning rate (`lr`) | `0.001` |
| Optimiseur | `adam` |
| Epochs | `50` |

F1 macro en validation croisée : **0.9948** — mais ce score est calculé **sur les données SMOTE**,
donc sur un jeu artificiellement équilibré. Il ne représente pas la performance réelle (0.568 sur le
test). C'est une illustration directe de l'importance d'évaluer sur un test **non rééquilibré**.

### Détail par classe sur le test

| Classe | Précision | Sensibilité | F1 | Nombre de cas |
|---|---|---|---|---|
| `No Failure` | 0.99 | 0.97 | 0.98 | 1 930 |
| `Overstrain Failure` | 0.79 | 0.94 | 0.86 | 16 |
| `Heat Dissipation Failure` | 0.68 | 0.86 | 0.76 | 22 |
| `Power Failure` | 0.83 | 0.79 | 0.81 | 19 |
| `Tool Wear Failure` | 0.00 | **0.00** | 0.00 | 9 |
| `Random Failures` | 0.00 | **0.00** | 0.00 | 4 |

### Interprétation

- **L'accuracy de 96,5 % est trompeuse** : elle vient presque entièrement de `No Failure`
  (1 930 bonnes réponses sur 2 000). Un modèle qui répondrait systématiquement « pas de panne »
  obtiendrait le même score. L'accuracy ne dit rien sur la capacité à détecter les pannes.
- **Le F1 macro de 0.568 révèle le vrai niveau** : il donne le même poids à chaque classe. L'écart
  entre 0.965 et 0.568 mesure exactement le pouvoir trompeur de l'accuracy.
- **Deux classes ne sont jamais détectées** : `Random Failures` (0/4) et `Tool Wear Failure` (0/9).
  Ce sont les deux classes les plus rares. Le modèle les range dans une autre classe plutôt que de
  les rater.
- Les classes **bien détectées** sont celles que l'EDA avait identifiées comme discriminantes :
  `Overstrain Failure` (0.94), `Heat Dissipation Failure` (0.86), `Power Failure` (0.79).
- **Train parfait (0.997) contre test à 0.568** : c'est le signe d'un **surapprentissage**, aggravé
  par SMOTE qui rend le train artificiellement « facile ».
- **Conclusion** : le modèle est utile mais incomplet. Il détecte les pannes fréquentes, mais les
  pannes rares lui échappent. La limite vient des **données** (4 et 9 exemples), pas de
  l'architecture.

### Régularisation

| Configuration | F1 macro (train) | F1 macro (test) | Précision (test) |
|---|---|---|---|
| Sans régularisation | 0.999 | 0.558 | 0.543 |
| **L2 (`weight_decay=1e-4`)** | 0.997 | **0.581** | 0.556 |
| Dropout 0.3 | 0.993 | 0.546 | 0.476 |
| Dropout + L2 | 0.992 | 0.551 | 0.485 |

**Interprétation :**

- **Le surapprentissage est net** : écart train/test de 0.997 contre 0.568, le plus grand du lab.
- **Le L2 est la meilleure configuration** (F1 test 0.581) : il réduit légèrement le train tout en
  améliorant le test — exactement le rôle attendu de la régularisation.
- **Le Dropout dégrade la précision** (0.476 contre 0.543) : il fait prédire trop souvent une panne
  rare. Il améliore un peu la sensibilité (0.669 contre 0.576), mais le compromis est défavorable.
- **Contrairement à la partie 1**, la régularisation **améliore réellement** le modèle
  (0.558 → 0.581). Il existait bien un surapprentissage à corriger.
- **Malgré cela**, le F1 test reste faible : aucune régularisation ne peut compenser l'absence de
  données sur `Random Failures` et `Tool Wear Failure`.

---

## 6. Choix des données et limites du travail

### Pourquoi `prices.csv` et pas les 3 autres fichiers du dataset NYSE ?

Le dataset NYSE fournit 4 fichiers. Un seul a été utilisé :

| Fichier | Écarté car |
|---|---|
| `prices-split-adjusted.csv` | **Même structure** que `prices.csv`, seulement corrigé des divisions d'actions. Ce serait un doublon. |
| `securities.csv` | Fiche descriptive des entreprises (nom, secteur, adresse) : **catégoriel, sans dimension temporelle**, aucun signal numérique pour un prix quotidien. |
| `fundamentals.csv` | Ratios comptables annuels (SEC 10-K, 2012-2016) : **granularité annuelle** alors que la régression porte sur des prix journaliers. Les joindre mélangerait deux fréquences. |

**Règle appliquée :** on retient le fichier qui contient **à la fois les variables d'entrée et la
cible, à la bonne fréquence**. `prices.csv` remplit ce critère.

Le second dataset (`predictive_maintenance.csv`) ne propose qu'**un seul fichier** : aucun choix à
faire. Il contient en revanche **deux cibles possibles** — `Target` (binaire : panne / pas panne) et
`Failure Type` (6 classes). L'énoncé demandant une classification **multi-classes**, c'est
`Failure Type` qui a été retenue comme cible, et `Target` a été supprimée des variables.

### Limites identifiées

**Partie 1 — régression**

1. **Un seul titre (AAPL).** L'énoncé ne le demande pas, mais travailler sur une seule série
   temporelle rend l'analyse lisible. Avec les 500 titres, il faudrait une autre stratégie (par
   exemple prédire le rendement plutôt que le prix).
2. **Prix non ajustés des divisions d'actions.** `prices.csv` est le fichier brut : le split 7:1
   d'Apple de juin 2014 apparaît comme une chute de −86 % artificielle. Le fichier
   `prices-split-adjusted.csv` fourni par l'auteur corrige cet effet — ce serait l'amélioration
   logique si le temps le permettait.
3. **Le modèle prédit la tendance, pas les mouvements.** Un R² de 0.939 signifie que le réseau
   retrouve le niveau du prix, pas sa variation quotidienne (impossible à anticiper : le marché est
   imprévisible à un jour).
4. **Validation croisée non chronologique.** `GridSearchCV` avec `cv=3` mélange les blocs d'une série
   temporelle, ce qui est légèrement optimiste pour la sélection des hyperparamètres. Le découpage
   **train/test principal reste chronologique**. `TimeSeriesSplit` serait plus rigoureux.

**Partie 2 — classification**

5. **SMOTE appliqué avant la validation croisée.** Les exemples synthétiques sont créés sur
   l'ensemble du train, puis le `GridSearchCV` redécoupe le résultat. Un exemple synthétique peut donc
   se retrouver dans un pli d'entraînement et son « parent » dans un pli de validation : c'est une
   fuite de données légère. La version rigoureuse utiliserait un `Pipeline` `imblearn` qui applique
   SMOTE **à l'intérieur** de chaque pli.
6. **Score de validation croisée non représentatif.** Le F1 macro de 0.9948 est obtenu sur des
   données SMOTE équilibrées. Le score réel sur le test non rééquilibré est de 0.568. Un lecteur
   pressé pourrait s'arrêter au premier chiffre : c'est le principal piège de ce lab.
7. **Deux classes indétectables.** `Random Failures` (18 cas) et `Tool Wear Failure` (45 cas) ont une
   sensibilité de 0. Avec si peu d'exemples, aucun modèle ne peut les apprendre de façon fiable —
   c'est une limite du **dataset**, pas de l'architecture.

---

## 7. Synthèse

**Ce que j'ai appris**

1. **PyTorch est explicite** : la boucle d'entraînement s'écrit à la main, en 4 étapes —
   `zero_grad` (effacer les gradients), *forward* (calculer la prédiction), `backward` (calculer les
   gradients), `step` (mettre à jour les poids). Rien n'est caché.
2. **Régression et classification ne se mesurent pas pareil** : MSE / RMSE / MAE / R² d'un côté,
   accuracy / précision / sensibilité / F1 de l'autre.
3. **L'ordre du temps compte** : sur une série temporelle, un découpage aléatoire entre train et test
   fausse complètement les résultats.
4. **Les données passent avant le modèle** : un dataset déséquilibré rend l'accuracy trompeuse, et
   aucun réseau ne peut compenser des classes trop rares.
5. **Les métriques doivent être choisies selon le problème** : sur un problème déséquilibré,
   le **F1 macro** et le **recall** sont plus informatifs que l'accuracy.
6. **Le surapprentissage se diagnostique** en comparant les courbes train et test. La régularisation
   sert à le corriger — mais elle **ne sert à rien s'il n'y en a pas** (partie 1) et **à elle seule ne
   suffit jamais** quand les données manquent (partie 2).
7. **Un score de validation croisée n'est pas un score de performance.** En partie 2, le F1 macro
   ressort à 0.9948 en validation croisée mais à 0.568 sur le test réel, parce que la validation
   portait sur des données SMOTE équilibrées. Il faut toujours évaluer sur un test qui ressemble à la
   réalité.
8. **Relier PyTorch à scikit-learn** demande une **enveloppe** (`fit` / `predict` / `get_params`) :
   c'est le prix à payer pour profiter de `GridSearchCV`, qui ne pilote que des objets scikit-learn.
9. **Les méthodes de rééchantillonnage réorganisent les données.** SMOTE range les exemples par
   classe : un `KFold` simple (sans mélange) peut alors produire un pli sans certaines classes. Il faut
   utiliser `StratifiedKFold`.

---

## 8. Reproductibilité

- Graine aléatoire fixée à `SEED = 42` (numpy et torch) → mêmes résultats à chaque exécution.
- Dépendances : voir `requirements.txt` à la racine du dépôt.

```bash
pip install -r requirements.txt
```

### Environnement utilisé

| Composant | Version |
|---|---|
| Python | 3.10+ (Colab) |
| PyTorch | 2.x |
| scikit-learn | 1.x |
| pandas / numpy / matplotlib / seaborn | récents |
| imbalanced-learn | ≥ 0.11 (pour SMOTE) |

---

## 9. Références

- PyTorch — <https://pytorch.org/tutorials/>
- scikit-learn — *Grid Search*, *Cross-validation*, *Metrics* — <https://scikit-learn.org/stable/>
- Chawla et al. (2002), *SMOTE: Synthetic Minority Over-sampling Technique*, JAIR.
- Dataset NYSE — <https://www.kaggle.com/datasets/dgawlik/nyse>
- Dataset Maintenance prédictive — <https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification>
