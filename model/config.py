"""
Configuration file for inceptionresnetv2, xception, vgg16, vgg19 
"""
import os

# Set project root directory
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Random seed for reproducibility
RANDOM_SEED = 123

#Model hyperparameters (v3)
MODEL = {
    'dense1_size': 2048,         # Dimension of first dense layer
    'dense2_size': 1024,         # Dimension of second dense layer
    'dropout': 0.45,             # Dropout probability
}

# Training hyperparameters
TRAINING = {
    'batch_size': 64,           # Batch size for training
    'learning_rate': 0.003,      # Learning rate for Adam
    'weight_decay': 0.0005,       # L2 regularization 
    'max_epochs': 30,           # Maximum number of training epochs
    'patience': 12,             # Early stopping patience
    'grad_clip': 1.0,           # Gradient clipping threshold
    'use_scheduler': True,      # Enable learning rate scheduling
    'scheduler_type': 'reduce_on_plateau',  # Reduce LR when validation metric plateaus
    'scheduler_factor': 0.5,    # Factor to reduce LR (multiply by 0.5)
    'scheduler_patience': 5,    # Wait 5 epochs before reducing LR
    'scheduler_min_lr': 1e-6,   # Minimum learning rate
}

# Data paths for vgg16_and_vgg19
v_n_v_DATA = {
    'data_dir' : os.path.join(ROOT, 'pre-processing', 'vgg16_and_vgg19', 'processed_data'), 
    'train_dir' : os.path.join(ROOT, 'pre-processing', 'vgg16_and_vgg19', 'processed_data', 'train'),  
    'val_dir' : os.path.join(ROOT, 'pre-processing', 'vgg16_and_vgg19', 'processed_data', 'val'),  
    'test_dir' : os.path.join(ROOT, 'pre-processing', 'vgg16_and_vgg19', 'processed_data', 'test') 
}

# Data paths for xception_and_resnet
x_n_r_DATA = {
    'data_dir' : os.path.join(ROOT, 'pre-processing', 'xception_and_resnet', 'processed_data'), 
    'train_dir' : os.path.join(ROOT, 'pre-processing', 'xception_and_resnet', 'processed_data', 'train'),  
    'val_dir' : os.path.join(ROOT, 'pre-processing', 'xception_and_resnet', 'processed_data', 'val'),  
    'test_dir' : os.path.join(ROOT, 'pre-processing', 'xception_and_resnet', 'processed_data', 'test') 
}


