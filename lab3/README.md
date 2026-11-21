# Atelier 3 — NLP arabe & génération de texte

Notebook unique : **[`lab3.ipynb`](lab3.ipynb)** (36 cellules, 19 cellules de code, **44 outputs conservés**, dont 3 figures).

Deux parties dans le même fichier, exécutées dans l’ordre sur **Google Colab / Tesla T4**, seed = 42 :

1. **Partie 1** — scraping arabe, preprocessing, embeddings AraBERT et comparaison **RNN / BiRNN / GRU / LSTM** ;
2. **Partie 2** — fine-tuning de **GPT-2 (124M)** sur un dataset maison et génération conditionnée.

Les données générées ne sont pas versionnées (voir `.gitignore`) : `data/*.csv`, `data/*.npz`, `data/arabert_emb.npy`, `trained_models/`, `data/generated_paragraphs.txt`.

---

# Partie 1 — NLP arabe avec architectures récurrentes

## 1. Collecte et score de pertinence

120 textes collectés par `requests` + `BeautifulSoup` :

| Source | Textes |
|---|---:|
| Wikipedia arabe | 35 |
| BBC Arabic | 35 |
| DW Arabic | 29 |
| France 24 Arabic | 21 |

Sujet : **الذكاء الاصطناعي**. Le score est déterministe :

```text
sim_s   = rang percentile(similarité TF-IDF char 2–4 grams avec le texte de référence)
hits_s  = min(nb mots-clés distincts, 6) / 6
score   = clip(1 + 6.5 * sim_s + 2.5 * hits_s, 0, 10)
```

Statistiques : **moyenne 4,47**, écart-type 2,07, min 1,10, max 10,00.

Discrétisation : `faible [0,4[`, `moyen [4,7[`, `élevé [7,10]` → **53 / 49 / 18** textes.

## 2. Pipeline NLP

Normalisation (diacritiques, tatweel, unification des lettres), tokenization regex, stop words, **ISRIStemmer**, lemmatization rule-based documentée, discrétisation. Vocabulaire final : **1 152 tokens** (freq ≥ 2, `PAD=0`, `UNK=1`), séquences `(120, 60)`, split **84 / 18 / 18**.

Pour le modèle, on utilise des embeddings contextuels **AraBERT** (`aubmindlab/bert-base-arabertv2`, 768 dims, figés) calculés sur le texte brut : le tokenizer sous-mots gère mieux la morphologie arabe qu’un stemming agressif.

## 3. Modèles et tuning

Architecture : `AraBERT (768) → Dropout → RNN / BiRNN / GRU / LSTM → Dropout → têtes régression + classification`.

Perte combinée : `5 * MSE(score/10) + CE(classe, poids inverses fréquences)` ; poids `[1.00, 1.10, 3.18]`.

Grille : `hidden_dim ∈ {64,128,256}`, `dropout ∈ {0.2,0.3,0.4}`, `lr ∈ {1e-3,5e-4}`, 30 epochs. 24 entraînements en **9,3 s** sur T4.

| Architecture | hidden_dim | dropout | lr | val_loss | Paramètres |
|---|---:|---:|---:|---:|---:|
| **BiRNN** | 256 | 0.3 | 5e-4 | **0.9268** | 527 364 |
| LSTM | 128 | 0.4 | 5e-4 | 1.0059 | 460 292 |
| RNN | 128 | 0.4 | 5e-4 | 1.0623 | 115 460 |
| GRU | 128 | 0.4 | 5e-4 | 1.1712 | 345 348 |

## 4. Résultats (test, 18 textes)

| Architecture | MAE ↓ | RMSE ↓ | R² ↑ | Accuracy ↑ | F1 macro ↑ | BLEU ↑ |
|---|---:|---:|---:|---:|---:|---:|
| **RNN** | 1.572 | 1.932 | 0.175 | **0.500** | **0.521** | **0.5000** |
| BiRNN | 1.913 | 2.290 | -0.160 | 0.500 | 0.386 | 0.5000 |
| GRU | 1.590 | **1.842** | **0.249** | 0.389 | 0.364 | 0.3889 |
| LSTM | 1.572 | 1.935 | 0.172 | 0.389 | 0.261 | 0.3889 |

Le RNN simple est le meilleur en classification (F1 0,521), le GRU en régression (R² 0,249). La BiRNN a la meilleure loss de validation mais sur-apprend sur ce petit test. BLEU est un **BLEU unigramme** sur les étiquettes discrétisées : il se confond avec l’accuracy et doit être lu comme tel. Le notebook contient les courbes de convergence et la matrice de confusion.

---

# Partie 2 — Transformer / Text generation avec GPT-2

`pytorch-transformers` est l’ancien nom déprécié : on installe **`transformers`** et on utilise `AutoTokenizer` / `AutoModelForCausalLM`, avec le tuto de [mf1024](https://gist.github.com/mf1024/3df214d2f17f3dcc56450ddf0d5a4cd7) comme référence.

## 1. Dataset maison

500 paragraphes anglais générés de façon déterministe (seed 42) par combinaison de templates sur l’IA (domaines, outils, acteurs, actions, métriques). Sauvegardé dans `data/custom_ai_paragraphs.csv`.

Chaque exemple est encodé `AI: <paragraphe> <|endoftext|>`, puis concaténé en blocs de **400 tokens** (astuce du tuto : on retire le premier token des exemples suivants) → **87 blocs**, longueur moyenne 363, split **79 train / 8 val**.

## 2. Fine-tuning

`gpt2` (124 439 808 params), AdamW lr 3e-5, weight decay 0.01, warmup linéaire 10 steps, accumulation 4, gradient clipping 1.0, 5 epochs = 100 optimizer steps. Durée : **285,6 s** sur T4. Meilleur modèle sauvegardé sur la validation.

| Epoch | Train loss | Train PPL | Val loss | Val PPL |
|---:|---:|---:|---:|---:|
| 1 | 3.0281 | 20.66 | 1.7216 | 5.59 |
| 2 | 1.4015 | 4.06 | 0.9429 | 2.57 |
| 3 | 0.9109 | 2.49 | 0.7616 | 2.14 |
| 4 | 0.7922 | 2.21 | 0.7224 | 2.06 |
| **5** | **0.7569** | **2.13** | **0.7102** | **2.03** |

## 3. Génération

Prompt : `AI: Artificial intelligence is transforming healthcare`. Échantillonnage top-k (`k=40`, `T=0.9`), 100 tokens max, arrêt sur `<|endoftext|>`.

| # | Tokens | Arrêt EOS | distinct-2 |
|---:|---:|:---:|---:|
| 1 | 89 | oui | 0.784 |
| 2 | 96 | oui | 0.874 |
| 3 | 63 | oui | 0.952 |

Extrait :

> *AI: Artificial intelligence is transforming healthcare more efficiently, but errors remain a challenge. A new deep learning system was deployed in healthcare to automate routine tasks for analysts. Researchers report that machine learning can personalize services more efficiently. While personalization helps designers to optimize routes more quickly, data privacy remains a challenge.*

Les 3 paragraphes sont sauvegardés dans `data/generated_paragraphs.txt`. Le modèle a appris le préfixe, le registre « rapport de recherche » et la structure ; le corpus reste synthétique et donc répétitif (limite documentée).

---

## Conclusion générale

Sur un même atelier, deux tâches NLP très différentes ont été traitées : une **classification/régression supervisée** sur 120 textes arabes (AraBERT + RNN, F1 0,521) et une **génération autorégressive** fine-tunée sur 500 paragraphes (perplexité de validation 5,59 → 2,03). Les deux pipelines sont reproductibles, avec seed fixée et données générées par les cellules du notebook.

## Reproduction

```python
!pip install -q transformers beautifulsoup4 requests nltk scikit-learn
```

Exécuter les 19 cellules de code de `lab3.ipynb` dans l’ordre ; au premier passage, AraBERT et GPT-2 sont téléchargés depuis Hugging Face.
