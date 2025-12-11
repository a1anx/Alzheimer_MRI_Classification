"""
Configuration file for inceptionresnetv2, xception, vgg16, vgg19 
"""

# Random seed for reproducibility
RANDOM_SEED = 123

# Model hyperparameters (v3)
MODEL = {
    'dense1_size': ,         # Dimension of token embeddings
    'dense2_size': 256,         # Dimension of token embeddings
    'dropout': 0.45,            # Dropout probability (initially 0.45)
    
}

# Training hyperparameters
TRAINING = {
    'batch_size': 64,           # Batch size for training
    'learning_rate': 0.0003,      # Learning rate for AdamW
    'weight_decay': 0.0005,       # L2 regularization 
    'max_epochs': 50,           # Maximum number of training epochs
    'patience': 10,             # Early stopping patience
    'grad_clip': 1.0,           # Gradient clipping threshold
    'use_scheduler': True,      # Enable learning rate scheduling
    'scheduler_type': 'reduce_on_plateau',  # Reduce LR when validation metric plateaus
    'scheduler_factor': 0.5,    # Factor to reduce LR (multiply by 0.5)
    'scheduler_patience': 5,    # Wait 5 epochs before reducing LR
    'scheduler_min_lr': 1e-6,   # Minimum learning rate
}
