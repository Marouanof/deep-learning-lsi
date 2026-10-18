# Deep Learning

Travaux pratiques de Deep Learning.

---

## Labs

| Lab | Sujet | Concepts clés | Rapport |
|---|---|---|---|
| **Lab 1** | **DNN / MLP avec PyTorch** — régression + classification multi-classes | `nn.Module`, boucle d'entraînement, `GridSearchCV`, Loss/Epochs, régularisation, SMOTE | [→ voir le rapport](lab1_pytorch_dnn/README.md) |

---

## Détail

### [Lab 1 — DNN / MLP avec PyTorch](lab1_pytorch_dnn/README.md)

Prise en main de **PyTorch** sur deux tâches d'apprentissage supervisé :

| Partie | Tâche | Dataset | Cible |
|---|---|---|---|
| 1 | **Régression** | [NYSE Stock Prices](https://www.kaggle.com/datasets/dgawlik/nyse) | prix de clôture de J+1 |
| 2 | **Classification multi-classes** | [Machine Predictive Maintenance](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification) | type de panne (`Failure Type`) |

Fichiers : `partie1_regression_nyse.ipynb`, `partie2_classification_maintenance.ipynb`

---

## Environnement

Les notebooks sont conçus pour **Google Colab** (Python 3.10+). Pour une exécution locale :

```bash
pip install -r requirements.txt
```

---

## Structure du dépôt

```text
deep-learning-lsi/
├── README.md              # ce fichier — index des labs
├── requirements.txt       # dépendances communes
├── .gitignore
└── lab1_pytorch_dnn/
    ├── Atelier1.pdf
    ├── README.md
    ├── partie1_regression_nyse.ipynb
    └── partie2_classification_maintenance.ipynb
```

Les **datasets ne sont pas versionnés** (voir `.gitignore`) : ils se téléchargent depuis Kaggle.