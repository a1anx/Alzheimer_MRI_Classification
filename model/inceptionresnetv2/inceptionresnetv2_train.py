import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os
from model_architecture.inceptionresnetv2_model import InceptionResNetV2_Finetune_Model
from ... import config

# Set random seed for reproducibility
tf.random.set_seed(config.RANDOM_SEED)

# Data paths
DATA_DIR = '/Users/andykim/Documents/2025 Fall/NNDL/Project/NNDL_MRI_PROJECT/split_augmented_data'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')
TEST_DIR = os.path.join(DATA_DIR, 'test')

# Image parameters
IMG_HEIGHT = 299
IMG_WIDTH = 299
BATCH_SIZE = config.TRAINING['batch_size']

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load data
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

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
model_wrapper = InceptionResNetV2_Finetune_Model(
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