# Deep Learning

Travaux pratiques de Deep Learning.

---

## Labs / Ateliers

| Atelier / Lab | Sujet | Concepts clés | Rapport |
|---|---|---|---|
| **Lab 1 (Atelier 1)** | **DNN / MLP avec PyTorch** — régression + classification multi-classes | `nn.Module`, boucle d'entraînement, `GridSearchCV`, Loss/Epochs, régularisation, SMOTE | [→ voir le rapport](lab1_pytorch_dnn/README.md) |
| **Lab 2 (Atelier 2)** | **Vision avec PyTorch** — CNN, Faster R-CNN, transfert (VGG16 / AlexNet), ViT | convolution, pooling, GPU mode, fine-tuning, ViT from scratch, comparaison de métriques | [→ voir le rapport](lab2_cnn_vit/README.md) |
| **Atelier 3 / Lab 3** | **NLP arabe & génération de texte** — Partie 1 RNN/AraBERT + Partie 2 GPT-2 | BeautifulSoup, TF-IDF, AraBERT, RNN / BiRNN / GRU / LSTM, GPT-2 124M, AdamW + warmup, top-k sampling, perplexité | [→ rapport](lab3/README.md) · [→ notebook unique](lab3/lab3.ipynb) |

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

### [Atelier 3 / Lab 3 — NLP arabe & génération de texte](lab3/README.md)

Un seul notebook, **[`lab3/lab3.ipynb`](lab3/lab3.ipynb)** (36 cellules, 44 outputs conservés dont 3 figures), exécuté sur T4 avec seed 42.

#### Partie 1 — NLP arabe avec architectures récurrentes

Scraping de 4 sites arabes autour de **الذكاء الاصطناعي** (120 textes), score 0–10 par TF-IDF, pipeline NLP complet, puis comparaison de **RNN / BiRNN / GRU / LSTM** sur des embeddings **AraBERT** figés. Tuning : 24 runs en **9,3 s**.

Résultats (test, 18 textes) : RNN **0,500 accuracy / 0,521 F1**, BiRNN 0,500 / 0,386, GRU 0,389 / 0,364 (meilleur R² à **0,249**), LSTM 0,389 / 0,261. Le BLEU unigramme se confond avec l’accuracy ; limites détaillées dans le [rapport](lab3/README.md).

#### Partie 2 — Transformer / Text generation avec GPT-2

Fine-tuning de **GPT-2 124M** sur 500 paragraphes maison (`AI: … <|endoftext|>`, 87 blocs de 400 tokens, 79 train / 8 val), AdamW + warmup, 5 epochs / 100 steps en **285,6 s**, puis génération top-k depuis `AI: Artificial intelligence is transforming healthcare`.

Résultats : loss de validation **1,7216 → 0,7102** sur 5 epochs (perplexité **5,59 → 2,03**), 3 paragraphes générés avec un `distinct-2` de 0,78–0,95 — extraits dans le [rapport](lab3/README.md).

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
├── lab2_cnn_vit/
│   ├── Atelier2.pdf
│   ├── README.md
│   ├── partie1_cnn_classifieur.ipynb
│   └── partie2_vit.ipynb
└── lab3/                        # Atelier 3 — parties 1 et 2
    ├── README.md
    └── lab3.ipynb               # notebook unique avec outputs conservés
```

Les **datasets ne sont pas versionnés** (voir `.gitignore`) : ceux des labs 1 et 2 se
téléchargent depuis Kaggle, ceux de l'**atelier 3** sont générés par les cellules de `lab3.ipynb`.