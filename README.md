# Deep Learning

Travaux pratiques de Deep Learning.

---

## Labs

| Lab | Sujet | Concepts clés | Rapport |
|---|---|---|---|
| **Lab 1** | **DNN / MLP avec PyTorch** — régression + classification multi-classes | `nn.Module`, boucle d'entraînement, `GridSearchCV`, Loss/Epochs, régularisation, SMOTE | [→ voir le rapport](lab1_pytorch_dnn/README.md) |
| **Lab 2** | **Vision avec PyTorch** — CNN, Faster R-CNN, transfert (VGG16 / AlexNet), ViT | convolution, pooling, GPU mode, fine-tuning, ViT from scratch, comparaison de métriques | [→ voir le rapport](lab2_cnn_vit/README.md) |

---

## Détail

### [Lab 1 — DNN / MLP avec PyTorch](lab1_pytorch_dnn/README.md)

Prise en main de **PyTorch** sur deux tâches d'apprentissage supervisé :

| Partie | Tâche | Dataset | Cible |
|---|---|---|---|
| 1 | **Régression** | [NYSE Stock Prices](https://www.kaggle.com/datasets/dgawlik/nyse) | prix de clôture de J+1 |
| 2 | **Classification multi-classes** | [Machine Predictive Maintenance](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification) | type de panne (`Failure Type`) |

Fichiers : `partie1_regression_nyse.ipynb`, `partie2_classification_maintenance.ipynb`

### [Lab 2 — Vision avec PyTorch](lab2_cnn_vit/README.md)

Classification d'images sur **[MNIST](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)** en mode **GPU** (Colab, Tesla T4) :

| Partie | Modèles | Questions |
|---|---|---|
| 1 | CNN construit avec PyTorch, Faster R-CNN, VGG16 et AlexNet pré-entraînés | 4 questions : architecture CNN, adaptation Faster R-CNN, comparaison des métriques (Accuracy, F1, Loss, temps de training), fine-tuning |
| 2 | Vision Transformer (ViT) construit **from scratch** | 2 questions : architecture ViT + classification MNIST, interprétation et comparaison avec la partie 1 |

Fichiers : `partie1_cnn_classifieur.ipynb`, `partie2_vit.ipynb`

Résultats (test) : CNN **0.9906** (68 s), Faster R-CNN **0.9940** (21.5 min), VGG16 0.9876, AlexNet 0.9890, ViT from scratch **0.9786** (176 s) — analyse détaillée dans le [rapport du lab](lab2_cnn_vit/README.md).

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
├── lab1_pytorch_dnn/
│   ├── Atelier1.pdf
│   ├── README.md
│   ├── partie1_regression_nyse.ipynb
│   └── partie2_classification_maintenance.ipynb
└── lab2_cnn_vit/
    ├── Atelier2.txt
    ├── README.md
    ├── partie1_cnn_classifieur.ipynb
    └── partie2_vit.ipynb
```

Les **datasets ne sont pas versionnés** (voir `.gitignore`) : ils se téléchargent depuis Kaggle.