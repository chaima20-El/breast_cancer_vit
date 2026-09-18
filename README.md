# Diagnostic du Cancer du Sein avec Vision Transformers

Projet de thèse sur l'élaboration de modèles d'apprentissage profond pour le diagnostic du cancer du sein utilisant des mammographies.

## 🎯 Objectif

Développer un système de classification d'images mammographiques utilisant Vision Transformers (ViT) pour détecter et classifier le cancer du sein.

## 🏗️ Architecture

- **Modèle**: Vision Transformer (ViT)
- **Framework**: TensorFlow 2.x
- **Type d'images**: Mammographies
- **Classes**: Bénin / Malin / Normal

## 📁 Structure du Projet

```
breast_cancer_vit/
├── data/
│   ├── raw/                    # Données brutes
│   ├── processed/              # Données prétraitées
│   └── augmented/              # Données augmentées
├── models/
│   ├── vit_model.py           # Architecture Vision Transformer
│   └── saved_models/          # Modèles entraînés
├── src/
│   ├── data_loader.py         # Chargement des données
│   ├── preprocessing.py       # Prétraitement
│   ├── augmentation.py        # Augmentation des données
│   ├── train.py               # Entraînement
│   ├── evaluate.py            # Évaluation
│   └── gradcam.py             # Visualisation Grad-CAM
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_results.ipynb
├── results/
│   ├── metrics/               # Métriques sauvegardées
│   ├── plots/                 # Graphiques
│   └── gradcam_viz/           # Visualisations Grad-CAM
├── config.py                  # Configuration
├── requirements.txt           # Dépendances
└── main.py                    # Script principal
```

## 📊 Métriques Implémentées

- Accuracy, Precision, Recall, F1-Score
- Sensibilité (Sensitivity/TPR)
- Spécificité (Specificity)
- AUC-ROC et courbe ROC
- Matrice de confusion
- Cohen's Kappa
- Matthews Correlation Coefficient (MCC)

## 🚀 Installation

```bash
pip install -r requirements.txt
```

## 💻 Utilisation

```bash
# Entraînement
python main.py --mode train --epochs 100 --batch_size 32

# Évaluation
python main.py --mode evaluate --model_path models/saved_models/best_model.h5

# Visualisation Grad-CAM
python main.py --mode gradcam --image_path data/test/sample.png
```

## 📝 Citation

Si vous utilisez ce code pour votre recherche, veuillez citer :

```
@phdthesis{votre_nom_2026,
  title={Élaboration de modèles d'apprentissage profond pour le diagnostic du cancer du sein},
  author={Votre Nom},
  year={2026},
  school={Votre Université}
}
```
