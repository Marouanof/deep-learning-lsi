# Lab 2 — Vision avec PyTorch : CNN, Faster R-CNN, transfert et ViT

**Dataset** : [MNIST Dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) (téléchargé automatiquement via `torchvision.datasets.MNIST` — le même jeu que celui du lien Kaggle de l'énoncé, 60 000 images d'entraînement / 10 000 de test, 28×28 en niveaux de gris, 10 classes).

**Mode d'exécution** : Google Colab, GPU **Tesla T4**, graine fixée à 42 pour la reproductibilité.

---

## 1. Objectif

Comparer des architectures de vision sur une même tâche — classer les chiffres manuscrits — en couvrant les 4 familles demandées par l'énoncé :

1. un **CNN from scratch** (PyTorch, `nn.Module`) ;
2. un **Faster R-CNN**, adapté à un problème de classification ;
3. des modèles **pré-entraînés ImageNet** appliqués à MNIST : **VGG16** et **AlexNet** ;
4. un **ViT from scratch** (partie 2).

Les métriques exigées : **Accuracy, F1, Loss, temps d'entraînement**.

## 2. Fichiers

| Fichier | Contenu |
|---|---|
| `partie1_cnn_classifieur.ipynb` | Questions 1 à 4 de la partie 1 (18 cellules, 4 figures) |
| `partie2_vit.ipynb` | Questions 1 et 2 de la partie 2 (10 cellules, 2 figures) |

Convention : une question de l'énoncé = une cellule de code, commentée en français.

## 3. Partie 1 — résultats

### Question 1 — CNN from scratch

Architecture : `Conv(1→32, 3×3) → MaxPool → Conv(32→64, 3×3) → MaxPool → Flatten → FC(3136→128) → Dropout(0.25) → FC(128→10)`.

Hyper-paramètres : Adam, lr 0.001, weight decay 1e-4, batch 128, **5 epochs**. 421 642 paramètres.

| Métrique | test |
|---|---|
| Accuracy | **0.9906** |
| F1 macro | 0.9906 |
| Loss | 0.0284 |
| Temps (GPU T4) | **68.1 s** |

La matrice de confusion montre que la confusion principale se situe entre `4` et `9` (~10 erreurs), les autres paires de classes étant sous les 8 erreurs.

### Question 2 — Faster R-CNN

Faster R-CNN est une architecture de **détection**, pas de classification. L'adaptation demandée (« faire la même chose ») est traitée littéralement : MNIST est vu comme un problème de détection — chaque image contient exactement un chiffre, on lui associe une boîte couvrant l'image, et on entraîne un vrai Faster R-CNN à prédire la boîte et la classe du chiffre.

Modèle : `fasterrcnn_mobilenet_v3_large_fpn` (torchvision, pré-entraîné COCO), tête remplacée pour **11 classes** (0 = arrière-plan, 1 à 10 = les chiffres). Image passée de 28×28 à 128×128. SGD lr 0.005, momentum 0.9, **3 epochs**. 18 917 458 paramètres.

| Métrique | test |
|---|---|
| Accuracy | **0.9940** |
| F1 macro | 0.9940 |
| Loss (somme des 4 termes du modèle) | 0.0395 |
| Temps | **1 293.3 s (~21.5 min)** |

Au déploiement, la prédiction du chiffre est la classe de la boîte avec le score maximal (1 est retiré à l'étiquette pour compenser l'arrière-plan).

### Question 3 — comparaison CNN vs Faster R-CNN

| Modèle | Accuracy | F1 | Loss | Temps |
|---|---|---|---|---|
| CNN from scratch | 0.9906 | 0.9906 | 0.0284 | 68.1 s |
| Faster R-CNN | 0.9940 | 0.9940 | 0.0395 | 1 293.3 s |

Le Faster R-CNN est **19× plus lent** à l'entraînement, pour un gain d'environ +0.3 point d'accuracy. Explication : le CNN traite un lot de 128 en un simple forward ; le Faster R-CNN ajoute le RPN, le FPN et l'assignation boîtes/étiquettes, avec un lot de 4 (les cibles de détection sont plus lourdes à construire). Sa puissance est taillée pour la détection de plusieurs objets, pas pour classer une image à un seul objet centré.

### Question 4 — transfert ImageNet (VGG16, AlexNet)

Protocole : images 28×28 → 224×224 (3 canaux), normalisation ImageNet, **backbone convolutif figé**, seule la tête de classification (10 sorties) est ré-entraînée, Adam lr 1e-4, **2 epochs**.

| Modèle | Accuracy | F1 | Loss | Temps | Params entraînables |
|---|---|---|---|---|---|
| VGG16 | 0.9876 | 0.9874 | 0.0534 | 851.8 s | 119 586 826 |
| AlexNet | 0.9890 | 0.9890 | 0.0344 | 147.5 s | 54 575 114 |

Interprétation : le transfert depuis ImageNet n'apporte rien d'avantageux ici — des images 28×28 en niveaux de gris sont étirées en 224×224 couleur, loin des features COCO/ImageNet ; le surcoût de calcul pèse plus lourd que le gain de score.

**Conclusion de la partie 1 :** toutes les architectures convergent vers 0.99, mais les coûts explosent. Pour une tâche simple, un petit CNN entraîné de zéro est le choix le plus raisonnable.

## 4. Partie 2 — ViT from scratch

### Question 1 — architecture et classification

ViT construit de zéro (~100 lignes de PyTorch), suivant la structure du [tutoriel de référence de l'énoncé](https://medium.com/mlearning-ai/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c), adapté à la taille de MNIST :

- **patchs** 4×4 via `Conv2d(1→64, kernel 4×4, stride 4)` → 49 patchs de dimension 64 ;
- jeton **CLS** appris, en tête de séquence ; embeddings de position appris (50 × 64) ;
- **4 blocs Transformer** : attention multi-têtes (4 têtes de 16), MLP interne 128 avec GELU, Dropout 0.1, pré-norm et connexions résiduelles ;
- tête de classification : le jeton CLS final → `Linear(64→10)`.

139 018 paramètres. Adam lr 0.001, weight decay 1e-4, batch 128, **10 epochs**.

| Métrique (test) | valeur |
|---|---|
| Accuracy | 0.9786 |
| F1 macro | 0.9784 |
| Loss | 0.0703 |
| Temps | 176.2 s |

### Question 2 — interprétation et comparaison avec la partie 1

| Modèle | Accuracy | F1 | Loss | Temps |
|---|---|---|---|---|
| CNN from scratch | 0.9906 | 0.9906 | 0.0284 | 68.1 s |
| Faster R-CNN | 0.9940 | 0.9940 | 0.0395 | 1 293.3 s |
| **ViT from scratch** | 0.9786 | 0.9784 | 0.0703 | 176.2 s |

Le ViT est **2.6× plus lent** que le CNN et **1.2 point derrière**. Pourquoi :

1. **Aucun prior de localité** : la convolution sait d'emblée que les pixels voisins comptent et qu'un motif reste le même en se déplaçant ; l'attention ne dispose d'aucun de ces biais et doit les apprendre depuis les données.
2. **Jeu de données trop petit pour l'attention** : les Transformers expriment leur puissance sur de très gros volumes (ViT s'est imposé d'abord sur ImageNet-21k) ; sur 60 000 images simples, le CNN atteint vite son plafond quand le ViT apprend encore sa grammaire visuelle.
3. **Convergence non saturée** à 10 epochs (l'accuracy test montait encore en fin d'entraînement) : plus d'epochs ou des patchs plus fins feraient probablement monter le score.

## 5. Incidents rencontrés et solutions

| Incident | Cause | Solution |
|---|---|---|
| Erreur CUDA `device-side assert` | tête de détection déclarée avec 2 classes alors que les labels vont de 1 à 10 (11 classes avec l'arrière-plan) | passer la tête à 11 classes ; le contexte CUDA corrompu a imposé un redémarrage du noyau et une réexécution complète |
| `module 'torchvision.models.detection' has no attribute 'fasterrcnn_mobilenet_v3_fpn'` | nom du modèle incorrect | le bon nom : `fasterrcnn_mobilenet_v3_large_fpn` |
| `Number of dimensions of repeat dims can not be smaller...` | tenseur 4D (batch, 1, 224, 224) passé à `repeat(3,1,1)` | `squeeze(0)` avant le `repeat` |
| Résultats très légèrement différents d'une réexécution à l'autre | graine fixée à 42, Dropout actif | seul l'ordre de tirage du Dropout varie → écarts de ~0.0006, sans conséquence |

## 6. Synthèse

1. Les **4 architectures font presque jeu égal sur MNIST (98.7 – 99.4 %)** : sur une tâche simple, le critère décisif n'est pas le score mais le **coût** — le petit CNN domine le compromis, le Faster R-CNN mobilise une puissance de détection inutile ici.
2. Le **transfert ImageNet n'a rien apporté sur MNIST** : le changement de format (28×28 gris → 224×224 couleur) et le temps d'entraînement pèsent plus lourd que le gain potentiel.
3. Le **ViT from scratch** se construit facilement en PyTorch, mais **sans prior de localité ni volume de données massif, il perd face au CNN** : les Transformers visuels brillent à grande échelle ; sur une petite base, le convolutif reste le plus rentable.
4. **Le temps d'entraînement est une métrique à part entière** : Accuracy, F1 et Loss ne suffisent pas pour arbitrer une architecture.
