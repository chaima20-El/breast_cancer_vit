"""
Configuration du projet
"""
import os

# Chemins
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
AUGMENTED_DATA_DIR = os.path.join(DATA_DIR, 'augmented')

MODELS_DIR = os.path.join(BASE_DIR, 'models')
SAVED_MODELS_DIR = os.path.join(MODELS_DIR, 'saved_models')

RESULTS_DIR = os.path.join(BASE_DIR, 'results')
METRICS_DIR = os.path.join(RESULTS_DIR, 'metrics')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
GRADCAM_DIR = os.path.join(RESULTS_DIR, 'gradcam_viz')

# Créer les répertoires s'ils n'existent pas
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, AUGMENTED_DATA_DIR,
                  MODELS_DIR, SAVED_MODELS_DIR, RESULTS_DIR, METRICS_DIR,
                  PLOTS_DIR, GRADCAM_DIR]:
    os.makedirs(directory, exist_ok=True)

# Paramètres du modèle Vision Transformer
IMAGE_SIZE = 224  # Taille des images d'entrée
PATCH_SIZE = 16   # Taille des patches
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2
PROJECTION_DIM = 768
NUM_HEADS = 12
TRANSFORMER_LAYERS = 12
MLP_HEAD_UNITS = [2048, 1024]

# Classes
NUM_CLASSES = 3
CLASS_NAMES = ['benign', 'malignant', 'normal']
CLASS_LABELS = {
    'benign': 0,
    'malignant': 1,
    'normal': 2
}

# Hyperparamètres d'entraînement
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
WARMUP_EPOCHS = 10
DROPOUT_RATE = 0.1

# Paramètres d'augmentation
AUGMENTATION_PROBABILITY = 0.8
ROTATION_RANGE = 15
WIDTH_SHIFT_RANGE = 0.1
HEIGHT_SHIFT_RANGE = 0.1
ZOOM_RANGE = 0.15
HORIZONTAL_FLIP = True
BRIGHTNESS_RANGE = [0.8, 1.2]
CONTRAST_RANGE = [0.8, 1.2]

# Split des données
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Early stopping et callbacks
EARLY_STOPPING_PATIENCE = 15
REDUCE_LR_PATIENCE = 7
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-7

# Seed pour reproductibilité
RANDOM_SEED = 42

# GPU
USE_MIXED_PRECISION = True
GPU_MEMORY_LIMIT = None  # None = utiliser toute la mémoire disponible
