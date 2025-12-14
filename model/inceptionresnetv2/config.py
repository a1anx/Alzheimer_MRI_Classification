"""
Configuration file for InceptionResNetV2
"""
import os

# Set project root directory - go up 2 levels from config.py to reach project root
# config.py is at: /model/inceptionresnetv2/config.py
# parent is /model/inceptionresnetv2, parent.parent is /model, parent.parent.parent is project root
_config_dir = os.path.dirname(os.path.abspath(__file__))  # /model/inceptionresnetv2
_model_dir = os.path.dirname(_config_dir)  # /model
ROOT = os.path.dirname(_model_dir)  # project root

# Random seed for reproducibility
RANDOM_SEED = 123

#Model hyperparameters (v3)
MODEL = {
    'dense1_size': 1024,         # Dimension of first dense layer
    'dense2_size': 1024,         # Dimension of second dense layer
    'dropout': 0.5,              # Dropout probability
}

# Training hyperparameters
TRAINING = {
    'batch_size': 64,           # Batch size for training
    'learning_rate': 0.003,    # Learning rate for Adam
    'max_epochs': 30,           # Maximum number of training epochs
    'patience': 12,             # Early stopping patience
    'scheduler_factor': 0.5,    # Factor to reduce LR (multiply by 0.5)
    'scheduler_patience': 5,    # Wait 5 epochs before reducing LR
    'scheduler_min_lr': 1e-7,   # Minimum learning rate
}

# Data paths for vgg16_and_vgg19
v_n_v_DATA = {
    'data_dir' : os.path.join(ROOT, 'preprocessing', 'vgg16_and_vgg19', 'processed_data'),
    'train_dir' : os.path.join(ROOT, 'preprocessing', 'vgg16_and_vgg19', 'processed_data', 'train'),
    'val_dir' : os.path.join(ROOT, 'preprocessing', 'vgg16_and_vgg19', 'processed_data', 'val'),
    'test_dir' : os.path.join(ROOT, 'preprocessing', 'vgg16_and_vgg19', 'processed_data', 'test')
}

# Data paths for xception_and_resnet
x_n_r_DATA = {
    'data_dir' : os.path.join(ROOT, 'preprocessing', 'xception_and_resnet', 'processed_data'),
    'train_dir' : os.path.join(ROOT, 'preprocessing', 'xception_and_resnet', 'processed_data', 'train'),
    'val_dir' : os.path.join(ROOT, 'preprocessing', 'xception_and_resnet', 'processed_data', 'val'),
    'test_dir' : os.path.join(ROOT, 'preprocessing', 'xception_and_resnet', 'processed_data', 'test')
}


