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

# Add parent directory path to import directories module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import directories as dir

# Label types 
label_types = {
    'NonDemented': 0,
    'VeryMildDemented': 1,
    'MildDemented': 2,
    'ModerateDemented': 3
}

# Resizing constant
resize_constant = (299, 299)

def count_class_distribution(dataset_path):
    """
    Count the number of images in each class category.

    Args:
        dataset_path: Path to dataset directory containing class subdirectories

    Returns:
        Dictionary with class counts and total count
    """
    class_counts = {class_name: 0 for class_name in label_types.keys()}

    for root, dirs, files in os.walk(dataset_path):
        # Get the class name from the directory structure
        for class_name in label_types.keys():
            if class_name in root:
                # Count only .jpeg files
                jpeg_files = [f for f in files if f.lower().endswith('.jpeg')]
                class_counts[class_name] += len(jpeg_files)
                break

    total_count = sum(class_counts.values())

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
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=5,
    zoom_range=0.05,
    horizontal_flip=True,
    fill_mode="nearest"
)

# Track processed images per class for verification
processed_counts = {class_name: 0 for class_name in label_types.keys()}

# Preprocess all train dataset
for root, dirs, files in os.walk(dir.TRAIN):
    for f in files:
        # Skip non-image files
        if not f.lower().endswith('.jpeg'):
            continue

        file_path = os.path.join(root, f)

        # Identify which class this image belongs to
        current_class = None
        for class_name in label_types.keys():
            if class_name in root:
                current_class = class_name
                break

        # Open image
        img = Image.open(file_path)

        # Convert to RGB
        rgb_img = img.convert("RGB")

        # Resize to 299x299 using Lanczos algorithm
        resized_img = rgb_img.resize(resize_constant, resample=Image.LANCZOS)

        # Convert to array (before normalization for augmentation)
        arr = np.array(resized_img, dtype=np.float32)

        # Save original image (with normalization)
        arr_original = (arr / 127.5) - 1.0
        save_path = os.path.join(processed_train_data, f"{os.path.splitext(f)[0]}.npy")
        np.save(save_path, arr_original)

        # Track this image
        if current_class:
            processed_counts[current_class] += 1

        # Generate augmented images that preserves original 4 class ratio
        arr_reshaped = arr.reshape((1,) + arr.shape)  # Add batch dimension for generator
        aug_iter = aug_data.flow(arr_reshaped, batch_size=1)

        for i in range(num_augmentations):
            # Generate augmented image
            aug_arr = next(aug_iter)[0]

            # Apply x_n_r normalization
            aug_arr_normalized = (aug_arr / 127.5) - 1.0

            # Save augmented image
            aug_save_path = os.path.join(processed_train_data, f"{os.path.splitext(f)[0]}_aug{i+1}.npy")
            np.save(aug_save_path, aug_arr_normalized)

            # Track augmented image (same class as original)
            if current_class:
                processed_counts[current_class] += 1

# Verify that augmentation preserved class ratio
print("After Augmentation:")
total_processed = sum(processed_counts.values())
for class_name, count in processed_counts.items():
    percentage = (count / total_processed * 100) if total_processed > 0 else 0
    original_percentage = (original_class_counts[class_name] / original_total * 100) if original_total > 0 else 0
    print(f"{class_name:20s}: {count:5d} images ({percentage:5.2f}%)")
print(f"{'TOTAL':20s}: {total_processed:5d} images")
print(f"Augmentation Factor: {total_processed / original_total:.2f}x" if original_total > 0 else "N/A")

# Verify ratio preservation
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



