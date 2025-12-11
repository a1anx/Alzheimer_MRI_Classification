import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os
import numpy as np
import sys
from pathlib import Path
from model_architecture.inceptionresnetv2_model import inceptionresnetv2

# Add parent directory path to import directories module 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Set seed from config file
tf.random.set_seed(config.RANDOM_SEED)

# Data paths
DATA_DIR = config.x_n_r_DATA['data_dir']
TRAIN_DIR = config.x_n_r_DATA['train_dir']
VAL_DIR = config.x_n_r_DATA['val_dir']
TEST_DIR = config.x_n_r_DATA['test_dir']

# Image parameters
IMG_HEIGHT = 299
IMG_WIDTH = 299
BATCH_SIZE = config.TRAINING['batch_size']

# Load all .npy files from directory and extract labels from filenames
def load_data(data_dir):
    images = []
    labels = []

    label_types = {
        'NonDemented' : 0,
        'VeryMildDemented' : 1,
        'MildDemented' : 2,
        'ModerateDemented' : 3
    }

    # Iterate over all 
    for file in Path(data_dir).glob('*.npy'):
        image = np.load(file)
        images.append(image)


val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Create model
print("Creating InceptionResNetV2 model...")
model_wrapper = inceptionresnetv2(
    dense1_size=config.MODEL['dense1_size'],
    dense2_size=config.MODEL['dense2_size'],
    dropout=config.MODEL['dropout']
)
model = model_wrapper.model

# Compile model
optimizer = tf.keras.optimizers.Adam(learning_rate=config.TRAINING['learning_rate'])
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=config.TRAINING['patience'],
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=config.TRAINING['scheduler_factor'],
        patience=config.TRAINING['scheduler_patience'],
        min_lr=config.TRAINING['scheduler_min_lr'],
        verbose=1
    ),
    ModelCheckpoint(
        'inceptionresnetv2_best.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# Train model
print("Starting training...")
history = model.fit(
    train_generator,
    epochs=config.TRAINING['max_epochs'],
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

# Evaluate on test set
print("\nEvaluating on test set...")
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Save final model
model.save('inceptionresnetv2_final.h5')
print("Model saved as inceptionresnetv2_final.h5")