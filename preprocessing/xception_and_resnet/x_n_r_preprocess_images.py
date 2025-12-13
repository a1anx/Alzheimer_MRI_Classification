"""
After running split_images.py,
Run this script ONCE to:
1) Count and display class distribution (NonDemented, VeryMildDemented, MildDemented, ModerateDemented)
2) Resize to match input dimensions required of xception and resnet (x_n_r) models (299,299)
3) Normalize MRI pixel values to range of (-1,1)
4) Perform data augmentation to increase size and diversity of train dataset by factor of 7
5) Verify that augmentation preserves original class ratio

INPUT:
Raw JPEG MRI images in dir.TRAIN, dir.VAL, dir.TEST.

OUTPUT:
Resized and normalized NumPy arrays saved in processed_data/train (with augmented variants),
processed_data/val, processed_data/test. Console output shows class distribution before and
after augmentation to verify ratio preservation.
"""

from PIL import Image
import numpy as np
import os
import sys
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Need to go up two directories to import the directories module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import directories as dir

# Map each dementia severity class to a numeric label
# These should match the label mappings used in the model training scripts
label_types = {
    'NonDemented': 0,
    'VeryMildDemented': 1,
    'MildDemented': 2,
    'ModerateDemented': 3
}

# Xception and ResNet models expect 299x299 input images
resize_constant = (299, 299)

def count_class_distribution(dataset_path):
    """
    Walks through the dataset directory and counts how many images belong to each class.
    Useful for checking if we have class imbalance in our data.
    """
    class_counts = {class_name: 0 for class_name in label_types.keys()}

    # Walk through all subdirectories in the dataset
    for root, dirs, files in os.walk(dataset_path):
        # Figure out which class this directory belongs to by checking the path
        for class_name in label_types.keys():
            if class_name in root:
                # Only count actual image files, not other random files
                jpeg_files = [f for f in files if f.lower().endswith('.jpeg')]
                class_counts[class_name] += len(jpeg_files)
                break

    total_count = sum(class_counts.values())

    # Print out the distribution nicely formatted
    for class_name, count in class_counts.items():
        percentage = (count / total_count * 100) if total_count > 0 else 0
        print(f"{class_name:20s}: {count:4d} images ({percentage:5.2f}%)")
    print(f"{'TOTAL':20s}: {total_count:4d} images")

    return class_counts, total_count

# Create directory for processed data
processed_data = os.path.join(dir.X_N_R, "processed_data")
os.makedirs(processed_data, exist_ok=True)

# train subfolder
processed_train_data = os.path.join(processed_data, "train") 
os.makedirs(processed_train_data, exist_ok=True)

# val subfolder
processed_val_data = os.path.join(processed_data, "val") 
os.makedirs(processed_val_data, exist_ok=True)

# test subfolder
processed_test_data = os.path.join(processed_data, "test") 
os.makedirs(processed_test_data, exist_ok=True)

# Count and display original class distribution before augmentation
print("Before Augmentation:")
original_class_counts, original_total = count_class_distribution(dir.TRAIN)

# Preprocess all train dataset with augmentation (number of augmented versions per image)
num_augmentations = 6

# Data augmentation generator
aug_data = ImageDataGenerator(
    rotation_range=10,              # rotate images up to 10 degrees
    width_shift_range=0.05,         # shift horizontally by up to 5%
    height_shift_range=0.05,        # shift vertically by up to 5%
    shear_range=5,                  # apply shear transformation
    zoom_range=0.05,                # zoom in/out by up to 5%
    horizontal_flip=True,           # randomly flip images horizontally
    fill_mode="nearest"             # fill in new pixels with nearest neighbor values
)

# Track processed images per class for verification
processed_counts = {class_name: 0 for class_name in label_types.keys()}

# Process all training images - both save original and create augmented versions
for root, dirs, files in os.walk(dir.TRAIN):
    for f in files:
        if not f.lower().endswith('.jpeg'):
            continue

        file_path = os.path.join(root, f)

        # Figure out the class by checking which label is in the directory path
        current_class = None
        for class_name in label_types.keys():
            if class_name in root:
                current_class = class_name
                break

        img = Image.open(file_path)
        rgb_img = img.convert("RGB")

        # Resize using Lanczos for better quality than simple bilinear
        resized_img = rgb_img.resize(resize_constant, resample=Image.LANCZOS)

        # Keep unnormalized version for augmentation, normalize later
        arr = np.array(resized_img, dtype=np.float32)

        # Normalize pixel values from [0, 255] to [-1, 1] range
        # This is what Xception and ResNet expect as input
        arr_original = (arr / 127.5) - 1.0
        save_path = os.path.join(processed_train_data, f"{os.path.splitext(f)[0]}.npy")
        np.save(save_path, arr_original)

        if current_class:
            processed_counts[current_class] += 1

        # Create augmented versions of this image
        # The generator needs a batch dimension, so reshape from (299, 299, 3) to (1, 299, 299, 3)
        arr_reshaped = arr.reshape((1,) + arr.shape)
        aug_iter = aug_data.flow(arr_reshaped, batch_size=1)

        for i in range(num_augmentations):
            aug_arr = next(aug_iter)[0]

            # Normalize the augmented image the same way
            aug_arr_normalized = (aug_arr / 127.5) - 1.0

            aug_save_path = os.path.join(processed_train_data, f"{os.path.splitext(f)[0]}_aug{i+1}.npy")
            np.save(aug_save_path, aug_arr_normalized)

            # Count this as part of the same class as the original
            if current_class:
                processed_counts[current_class] += 1

# Show the class distribution after augmentation and verify it matches the original ratios
print("After Augmentation:")
total_processed = sum(processed_counts.values())
for class_name, count in processed_counts.items():
    percentage = (count / total_processed * 100) if total_processed > 0 else 0
    original_percentage = (original_class_counts[class_name] / original_total * 100) if original_total > 0 else 0
    print(f"{class_name:20s}: {count:5d} images ({percentage:5.2f}%)")
print(f"{'TOTAL':20s}: {total_processed:5d} images")
print(f"Augmentation Factor: {total_processed / original_total:.2f}x" if original_total > 0 else "N/A")

# Double-check that each class maintained its proportion
# Since each image gets the same number of augmentations, the ratios should be identical
print("\nRatio Verification:")
ratio_preserved = True
for class_name in label_types.keys():
    original_ratio = original_class_counts[class_name] / original_total if original_total > 0 else 0
    augmented_ratio = processed_counts[class_name] / total_processed if total_processed > 0 else 0
    ratio_diff = abs(original_ratio - augmented_ratio)
    status = "Preserved" if ratio_diff < 0.001 else "Changed"
    print(f"{class_name:20s}: {status}")

# Preprocess all val dataset
for root, dirs, files in os.walk(dir.VAL):
    for f in files:
        # Skip non-image files
        if not f.lower().endswith('.jpeg'):
            continue

        file_path = os.path.join(root, f)

        # Open image
        img = Image.open(file_path)

        # Convert to RGB
        rgb_img = img.convert("RGB")

        # Resize to 299x299 using Lanczos algorithm
        resized_img = rgb_img.resize(resize_constant, resample=Image.LANCZOS)

        # Normalize images to range [-1, 1]
        arr = np.array(resized_img, dtype=np.float32)
        arr = (arr / 127.5) - 1.0

        # Save as NumPy arrays
        save_path = os.path.join(processed_val_data, f"{os.path.splitext(f)[0]}.npy")
        np.save(save_path, arr)

# Preprocess all test dataset
for root, dirs, files in os.walk(dir.TEST):
    for f in files:
        # Skip non-image files
        if not f.lower().endswith('.jpeg'):
            continue

        file_path = os.path.join(root, f)

        # Open image
        img = Image.open(file_path)

        # Convert to RGB
        rgb_img = img.convert("RGB")

        # Resize to 299x299 using Lanczos algorithm
        resized_img = rgb_img.resize(resize_constant, resample=Image.LANCZOS)

        # Normalize images to range [-1, 1]
        arr = np.array(resized_img, dtype=np.float32)
        arr = (arr / 127.5) - 1.0

        # Save as NumPy arrays
        save_path = os.path.join(processed_test_data, f"{os.path.splitext(f)[0]}.npy")
        np.save(save_path, arr)



