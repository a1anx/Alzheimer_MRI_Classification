import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os
import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from datetime import datetime

# Go up two directories to access the config file 
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

from inceptionresnetv2_model import inceptionresnetv2

# Set random seed for reproducibility across runs
tf.random.set_seed(config.RANDOM_SEED)

# Pull data paths from config
DATA_DIR = config.x_n_r_DATA['data_dir']
TRAIN_DIR = config.x_n_r_DATA['train_dir']
VAL_DIR = config.x_n_r_DATA['val_dir']
TEST_DIR = config.x_n_r_DATA['test_dir']

# InceptionResNetV2 expects 299x299 images
IMG_HEIGHT = 299
IMG_WIDTH = 299
BATCH_SIZE = config.TRAINING['batch_size']

# Custom data generator that loads batches on-the-fly instead of loading everything into memory
# This is necessary because the augmented training set is too large to fit in RAM
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_dir, batch_size=64, shuffle=True):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Same label mapping as preprocessing script
        self.label_types = {
            'NonDemented': 0,
            'VeryMildDemented': 1,
            'MildDemented': 2,
            'ModerateDemented': 3
        }

        # Get all .npy file paths and create shuffleable indexes
        self.file_paths = list(Path(data_dir).glob('*.npy'))
        self.indexes = np.arange(len(self.file_paths))

        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        # Number of batches per epoch
        return int(np.ceil(len(self.file_paths) / self.batch_size))

    def __getitem__(self, index):
        # Get the file indexes for this batch
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_files = [self.file_paths[i] for i in batch_indexes]

        images = []
        labels = []

        # Load each file in the batch
        for file in batch_files:
            image = np.load(file)
            images.append(image)

            # Extract the label from the filename (e.g., "NonDemented_123.npy")
            filename = file.stem
            for label_name, label_value in self.label_types.items():
                if label_name in filename:
                    labels.append(label_value)
                    break

        images = np.array(images, dtype=np.float32)
        # Convert labels to one-hot encoding
        labels = tf.keras.utils.to_categorical(labels, num_classes=4)

        return images, labels

    def on_epoch_end(self):
        # Reshuffle after each epoch if shuffle is enabled
        if self.shuffle:
            np.random.shuffle(self.indexes)

# Test set is small enough to load everything into memory at once
# This makes final evaluation faster since we don't need to load files repeatedly
def load_test_data(data_dir):
    images = []
    labels = []

    label_types = {
        'NonDemented': 0,
        'VeryMildDemented': 1,
        'MildDemented': 2,
        'ModerateDemented': 3
    }

    for file in Path(data_dir).glob('*.npy'):
        image = np.load(file)
        images.append(image)

        # Parse label from filename
        filename = file.stem
        for label, value in label_types.items():
            if label in filename:
                labels.append(value)
                break

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    return images, labels

# Set up generators for training and validation
# Shuffle training data for better learning, but not validation (we want consistent val metrics)
train_generator = DataGenerator(TRAIN_DIR, batch_size=BATCH_SIZE, shuffle=True)
val_generator = DataGenerator(VAL_DIR, batch_size=BATCH_SIZE, shuffle=False)

# Print dataset sizes
print(f"Training samples: {len(train_generator.file_paths)} ({len(train_generator)} batches)")
print(f"Validation samples: {len(val_generator.file_paths)} ({len(val_generator)} batches)")

# Load test data into memory
X_test, y_test = load_test_data(TEST_DIR)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=4)
print(f"Test samples: {len(X_test)}")
print(f"Test data shape: {X_test.shape}, Labels shape: {y_test.shape}")

# Build the model with hyperparameters from config
model_wrapper = inceptionresnetv2(
    dense1_size=config.MODEL['dense1_size'],
    dense2_size=config.MODEL['dense2_size'],
    dropout=config.MODEL['dropout']
)
model = model_wrapper.model

# Compile with Adam optimizer and categorical crossentropy for multi-class classification
optimizer = tf.keras.optimizers.Adam(learning_rate=config.TRAINING['learning_rate'])
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Create a timestamped output directory so multiple runs don't overwrite each other
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
run_output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model_output', f'run_{timestamp}')
checkpoint_dir = os.path.join(run_output_dir, 'checkpoints')
os.makedirs(checkpoint_dir, exist_ok=True)

# Set up training callbacks for better training control
callbacks = [
    # Stop training if validation loss doesn't improve for 'patience' epochs
    EarlyStopping(
        monitor='val_loss',
        patience=config.TRAINING['patience'],
        restore_best_weights=True
    ),
    # Reduce learning rate when validation loss plateaus
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    ),
    # Save the best model checkpoint based on validation accuracy
    ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, 'epoch_{epoch:02d}_val_acc_{val_accuracy:.4f}.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
]

# Train the model
history = model.fit(
    train_generator,
    epochs=config.TRAINING['max_epochs'],
    validation_data=val_generator,
    callbacks=callbacks
)

# Evaluate on test set to get final performance metrics
test_loss, test_accuracy = model.evaluate(X_test, y_test)

# Count model parameters to understand model complexity
trainable_params = sum([tf.size(var).numpy() for var in model.trainable_variables])
non_trainable_params = sum([tf.size(var).numpy() for var in model.non_trainable_variables])
total_params = trainable_params + non_trainable_params

print(f"\nModel Parameters:")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Non-trainable parameters: {non_trainable_params:,}")
print(f"Total parameters: {total_params:,}")

print(f"\nTest Loss (CCE): {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Get predictions and convert from one-hot to class indices
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Calculate precision, recall, F1 weighted by class frequency
precision, recall, f1, _ = precision_recall_fscore_support(y_true_classes, y_pred_classes, average='weighted')
print(f"\nWeighted Metrics:")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Show detailed per-class metrics
class_names = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
print(f"\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))

# Confusion matrix shows where the model is making mistakes
cm = confusion_matrix(y_true_classes, y_pred_classes)
print(f"\nConfusion Matrix:")
print(cm)

# Save metrics to file
metrics_path = os.path.join(run_output_dir, 'inceptionresnetv2_metrics.txt')
with open(metrics_path, 'w') as f:
    f.write("InceptionResNetV2 Model Evaluation Metrics\n")
    f.write("=" * 50 + "\n\n")

    # Write model architecture summary
    f.write("Model Architecture Summary:\n")
    f.write("-" * 50 + "\n")
    model.summary(print_fn=lambda x: f.write(x + '\n'))
    f.write("\n" + "=" * 50 + "\n\n")

    # Write configuration parameters
    f.write("Configuration Parameters:\n")
    f.write("-" * 50 + "\n")
    f.write(f"Random Seed: {config.RANDOM_SEED}\n\n")

    # Write model hyperparameters
    f.write("Model Hyperparameters:\n")
    f.write(f"  Dense Layer 1 Size: {config.MODEL['dense1_size']}\n")
    f.write(f"  Dense Layer 2 Size: {config.MODEL['dense2_size']}\n")
    f.write(f"  Dropout Rate: {config.MODEL['dropout']}\n\n")

    # Write training hyperparameters
    f.write("Training Hyperparameters:\n")
    f.write(f"  Batch Size: {config.TRAINING['batch_size']}\n")
    f.write(f"  Learning Rate: {config.TRAINING['learning_rate']}\n")
    f.write(f"  Max Epochs: {config.TRAINING['max_epochs']}\n")
    f.write(f"  Early Stopping Patience: {config.TRAINING['patience']}\n")
    f.write(f"  LR Scheduler Factor: {config.TRAINING['scheduler_factor']}\n")
    f.write(f"  LR Scheduler Patience: {config.TRAINING['scheduler_patience']}\n")
    f.write(f"  Min Learning Rate: {config.TRAINING['scheduler_min_lr']}\n\n")
    f.write("=" * 50 + "\n\n")

    # Write model parameters
    f.write("Model Parameters:\n")
    f.write(f"Trainable parameters: {trainable_params:,}\n")
    f.write(f"Non-trainable parameters: {non_trainable_params:,}\n")
    f.write(f"Total parameters: {total_params:,}\n\n")
    f.write(f"Test Loss (CCE): {test_loss:.4f}\n")
    f.write(f"Test Accuracy: {test_accuracy:.4f}\n\n")
    f.write("Weighted Metrics:\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1-Score: {f1:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_true_classes, y_pred_classes, target_names=class_names))
    f.write("\n\nConfusion Matrix:\n")
    f.write(str(cm))
print(f"Metrics saved to {metrics_path}")

# Save the trained model for later use
model_path = os.path.join(run_output_dir, 'inceptionresnetv2_final.h5')
model.save(model_path)
print(f"\nModel saved to {model_path}")

# Plot training history to visualize learning progress
plt.figure(figsize=(10, 6))

plt.plot(history.history['loss'], label='Train Loss', linestyle='-', linewidth=2)
plt.plot(history.history['val_loss'], label='Val Loss', linestyle='-', linewidth=2)
plt.plot(history.history['accuracy'], label='Train Accuracy', linestyle='--', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Val Accuracy', linestyle='--', linewidth=2)

plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Training and Validation Metrics')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

curves_path = os.path.join(run_output_dir, 'inceptionresnetv2_training_curves.png')
plt.savefig(curves_path)
plt.close()
print(f"Training curves saved to: {curves_path}")

# Create a visual confusion matrix to see where misclassifications happen
plt.figure(figsize=(8, 6))
im = plt.imshow(cm, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(range(4), class_names, rotation=45, ha='right')
plt.yticks(range(4), class_names)
# Add count numbers to each cell
for i in range(4):
    for j in range(4):
        plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i, j] > cm.max()/2 else 'black')
plt.colorbar(im)
plt.tight_layout()
matrix_path = os.path.join(run_output_dir, 'inceptionresnetv2_confusion_matrix.png')
plt.savefig(matrix_path)
plt.close()
print(f"Confusion matrix saved to {matrix_path}")
print(f"All outputs saved to {run_output_dir}")
