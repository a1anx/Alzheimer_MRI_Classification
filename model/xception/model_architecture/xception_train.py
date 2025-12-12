import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os
import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from xception_model import Xception

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

# Load all .npy files from  and extract labels from filenames
def load_data(data_dir):
    images = []
    labels = []

    label_types = {
        'NonDemented' : 0,
        'VeryMildDemented' : 1,
        'MildDemented' : 2,
        'ModerateDemented' : 3
    }

    # Iterate over all .npy files in the dir
    for file in Path(data_dir).glob('*.npy'):
        image = np.load(file)
        images.append(image)

        # Extract labels from filename 
        filename = file.stem
        for label, value in label_types.items():
            if label in filename:
                labels.append(value)
                break
    
    # Convert to numpy
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    return images, labels

# Load train data
X_train, y_train = load_data(TRAIN_DIR)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=4)
print(f"Train data shape: {X_train.shape}, Labels shape: {y_train.shape}")

# Load val data
X_val, y_val = load_data(VAL_DIR)
y_val = tf.keras.utils.to_categorical(y_val, num_classes=4)
print(f"Val data shape: {X_val.shape}, Labels shape: {y_val.shape}")

# Load test data
X_test, y_test = load_data(TEST_DIR)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=4)
print(f"Test data shape: {X_test.shape}, Labels shape: {y_test.shape}")

#
model_wrapper = xception(
    dense1_size=config.MODEL['dense1_size'],
    dense2_size=config.MODEL['dense2_size'],
    dropout=config.MODEL['dropout']
)
model = model_wrapper.model

optimizer = tf.keras.optimizers.Adam(learning_rate=config.TRAINING['learning_rate'])
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

checkpoint_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'model_output', 'checkpoints')
os.makedirs(checkpoint_dir, exist_ok=True)

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=config.TRAINING['patience'],
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    ),
    ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, 'epoch_{epoch:02d}_val_acc_{val_accuracy:.4f}.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
]

history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=config.TRAINING['max_epochs'],
    validation_data=(X_val, y_val),
    callbacks=callbacks
)

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Loss (CCE): {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

precision, recall, f1, _ = precision_recall_fscore_support(y_true_classes, y_pred_classes, average='weighted')
print(f"\nWeighted Metrics:")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

class_names = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
print(f"\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))

cm = confusion_matrix(y_true_classes, y_pred_classes)
print(f"\nConfusion Matrix:")
print(cm)

output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'model_output')
os.makedirs(output_dir, exist_ok=True)

model_path = os.path.join(output_dir, 'xception_final.h5')
model.save(model_path)
print(f"\nModel saved to: {model_path}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(history.history['loss'], label='train')
axes[0].plot(history.history['val_loss'], label='val')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training and Validation Loss')
axes[0].legend()

axes[1].plot(history.history['accuracy'], label='train')
axes[1].plot(history.history['val_accuracy'], label='val')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Training and Validation Accuracy')
axes[1].legend()

plt.tight_layout()
curves_path = os.path.join(output_dir, 'xception_training_curves.png')
plt.savefig(curves_path)
plt.close()
print(f"Training curves saved to: {curves_path}")

plt.figure(figsize=(8, 6))
im = plt.imshow(cm, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(range(4), class_names, rotation=45, ha='right')
plt.yticks(range(4), class_names)
for i in range(4):
    for j in range(4):
        plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i, j] > cm.max()/2 else 'black')
plt.colorbar(im)
plt.tight_layout()
matrix_path = os.path.join(output_dir, 'xception_confusion_matrix.png')
plt.savefig(matrix_path)
plt.close()
print(f"Confusion matrix saved to: {matrix_path}")
