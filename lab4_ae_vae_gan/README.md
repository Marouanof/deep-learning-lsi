# Lab 4 — AE, VAE et GANs avec PyTorch

**Datasets** :
- Partie 1 : [MNIST Dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) (téléchargé automatiquement via `torchvision.datasets.MNIST` — 60 000 images d'entraînement / 10 000 de test, 28×28 en niveaux de gris, 10 classes).
- Partie 2 : [Abstract Art Gallery](https://www.kaggle.com/datasets/bryanb/abstract-art-gallery) (téléchargement manuel via Kaggle, images redimensionnées en 64×64 RGB).

**Mode d'exécution** : Google Colab, GPU Tesla T4, graine fixée à 42.

---

## 1. Objectif

Couvrir les deux familles de modèles génératifs demandées par l'énoncé :

1. un **Auto-Encodeur (AE)** et un **Variational Auto-Encodeur (VAE)** sur MNIST, avec comparaison des pertes (MSE, KL divergence), des reconstructions et de l'espace latent ;
2. un **GAN (DCGAN)** sur l'Abstract Art Gallery, avec génération de nouvelles œuvres et comparaison avec les originales.

## 2. Fichiers

| Fichier | Contenu |
|---|---|
| `partie1_ae_vae.ipynb` | AE + VAE sur MNIST : architectures, entraînement, loss, reconstructions, espace latent 2D, génération |
| `partie2_gan.ipynb` | DCGAN sur Abstract Art : Generator/Discriminator, loss, entraînement, courbes, génération |

Convention : une question de l'énoncé = une cellule de code, commentée en français.

## 3. Partie 1 — AE et VAE sur MNIST

### Hyperparamètres communs

| Paramètre | Valeur |
|---|---|
| Espace latent | 2 (traçable directement) |
| Batch size | 128 |
| Epochs | 15 |
| Optimiseur | Adam, lr 1e-3 |
| Loss AE | MSE |
| Loss VAE | MSE (reconstruction) + KL divergence (β=1) |

### Architecture

- **Encodeur** : `Conv(1→32, 3×3, stride 2) → ReLU → Conv(32→64, 3×3, stride 2) → ReLU → Flatten → Linear(64·7·7 → 2)`.
- **Décodeur** : `Linear(2 → 64·7·7) → Unflatten → ConvTranspose(64→32) → ReLU → ConvTranspose(32→1) → Sigmoid`.
- **VAE** : l'encodeur produit μ et log σ² ; reparamétrisation z = μ + σ·ε ; même décodeur.

### Résultats attendus

| Modèle | Reconstruction | Génération | Espace latent |
|---|---|---|---|
| AE | très bonne (MSE faible) | non (pas de prior) | clusters séparés, trous |
| VAE | bonne (MSE un peu plus élevée) | oui (z ~ N(0,I)) | nuage continu et régulier |

### Évaluation et conclusion

- **Loss** : l'AE minimise uniquement la reconstruction et obtient une MSE slightly plus faible ; le VAE ajoute la KL qui régularise l'espace latent au prix d'une reconstruction un peu moins nette.
- **KL divergence** : elle mesure l'écart entre q(z|x) et le prior N(0,I). Elle force la continuité de l'espace latent et rend la génération possible.
- **Espace latent** : AE → clusters séparés, trous dans l'espace (pas de génération fiable) ; VAE → nuage continu et régulier, génération et interpolation possibles.
- **Choix** : utiliser l'AE pour la compression/reconstruction pure, le VAE pour la génération et l'interpolation dans l'espace latent.

## 4. Partie 2 — GAN sur Abstract Art Gallery

### Hyperparamètres DCGAN

| Paramètre | Valeur |
|---|---|
| Taille d'image | 64×64 (RGB) |
| Dimension du bruit z | 100 |
| Filtres G / D | 64 |
| Batch size | 64 |
| Epochs | 50 |
| Optimiseur | Adam, lr 2e-4, β1=0.5 |
| Loss | BCE (binaire croisée) |

### Architecture

- **Générateur** : z (100,1,1) → 5 blocs `ConvTranspose2d` + BatchNorm + ReLU → Tanh, sortie 64×64×3 dans [-1,1].
- **Discriminateur** : 5 blocs `Conv2d` + BatchNorm + LeakyReLU → Sigmoid, sortie scalaire (réel/faux).

### Évaluation et conclusion

- **Loss** : la loss du Discriminateur reste autour de ~0.5–1.0 et celle du Générateur oscille ; un équilibre (ni effondrement du G, ni D trop fort) indique un entraînement stable.
- **KL divergence** : **non applicable** aux GANs — ils n'optimisent pas de borne variationnelle et n'imposent pas de prior explicite dans la loss. On suit les courbes de loss D/G et la qualité visuelle.
- **Qualité** : les images générées capturent le style abstrait (couleurs, textures) mais peuvent souffrir de mode collapse et de flou ; plus d'epochs / tuning (lr, architecture) améliore le résultat.
- **Choix** : DCGAN simple et efficace pour démarrer ; WGAN-GP ou StyleGAN pour une meilleure stabilité/qualité.

## 5. Incidents possibles et solutions

| Incident | Cause | Solution |
|---|---|---|
| `FileNotFoundError` sur le dossier d'images | chemin `DATA_ROOT` incorrect ou dataset non décompressé | vérifier le chemin et décompresser le zip Kaggle |
| Mode collapse (images toutes identiques) | déséquilibre G/D | réduire le lr du D, ajouter du bruit, augmenter les epochs |
| Sortie saturée (tout noir/blanc) | mauvaise normalisation | garder Normalize(0.5,0.5) et Tanh en sortie du G |
| CUDA out of memory | batch trop grand pour le GPU | réduire `BATCH_SIZE` |

## 6. Synthèse

1. L'**AE** excelle en reconstruction mais son espace latent n'est pas régularisé : pas de génération fiable.
2. Le **VAE** régularise l'espace latent via la KL divergence et permet la génération et l'interpolation, au prix d'une reconstruction légèrement moins nette.
3. Le **GAN** apprend implicitement la distribution des données et génère des images réalistes, mais est plus instable à entraîner et ne fournit pas de métrique de vraisemblance (pas de KL).
4. Le choix du modèle dépend de l'objectif : reconstruction → AE ; génération avec espace latent structuré → VAE ; génération haute fidélité → GAN.
