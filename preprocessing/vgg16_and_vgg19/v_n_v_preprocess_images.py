"""
After running split_images.py,
Run this script ONCE to:
1) Resize to match input dimensions required of VGG16 and VGG19 models (224,224)
2) Normalize MRI pixel values using ImageNet mean subtraction
3) Perform data augmentation to increase size and diversity of train dataset by factor of 6

INPUT:
Raw JPEG MRI images in dir.TRAIN, dir.VAL, dir.TEST.

OUTPUT:
Resized and normalized NumPy arrays saved in processed_data/train (with augmented variants), processed_data/val, processed_data/test.
"""

from PIL import Image
import numpy as np
import os
import sys
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Add parent directory to path to import directories module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import directories as dir

# Resizing constant 
resize_constant = (224, 224)

# Create directory for processed data
processed_data = os.path.join(dir.V_N_V, "processed_data")
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

for root, dirs, files in os.walk(dir.TRAIN):
    for f in files:
        # Skip non-image files
        if not f.lower().endswith('.jpeg'):
            continue

        file_path = os.path.join(root, f)

        # Open image
        img = Image.open(file_path)

        # Convert to RGB
        rgb_img = img.convert("RGB")

        # Resize to 224x224 using Lanczos algorithm
        resized_img = rgb_img.resize(resize_constant, resample=Image.LANCZOS)

        # Convert to array (before normalization for augmentation)
        arr = np.array(resized_img, dtype=np.float32)

        # Save original image (with normalization)
        arr_original = arr.copy()
        arr_original[:,:,0] -= 103.939  # Blue
        arr_original[:,:,1] -= 116.779  # Green
        arr_original[:,:,2] -= 123.68   # Red
        save_path = os.path.join(processed_train_data, f"{os.path.splitext(f)[0]}.npy")
        np.save(save_path, arr_original)

        # Generate augmented versions
        arr_reshaped = arr.reshape((1,) + arr.shape)  # Add batch dimension for generator
        aug_iter = aug_data.flow(arr_reshaped, batch_size=1)

        for i in range(num_augmentations):
            # Generate augmented image
            aug_arr = next(aug_iter)[0]

            # Apply VGG normalization
            aug_arr[:,:,0] -= 103.939  # Blue
            aug_arr[:,:,1] -= 116.779  # Green
            aug_arr[:,:,2] -= 123.68   # Red

            # Save augmented image
            aug_save_path = os.path.join(processed_train_data, f"{os.path.splitext(f)[0]}_aug{i+1}.npy")
            np.save(aug_save_path, aug_arr)

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

        # Resize to 224x224 using Lanczos algorithm
        resized_img = rgb_img.resize(resize_constant, resample=Image.LANCZOS)

        # Convert to array and subtract ImageNet means (BGR for VGG)
        arr = np.array(resized_img, dtype=np.float32)
        arr[:,:,0] -= 103.939  # Blue 
        arr[:,:,1] -= 116.779  # Green 
        arr[:,:,2] -= 123.68   # Red 

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

        # Resize to 224x224 using Lanczos algorithm
        resized_img = rgb_img.resize(resize_constant, resample=Image.LANCZOS)

        # Convert to array and subtract ImageNet means (BGR for VGG)
        arr = np.array(resized_img, dtype=np.float32)
        arr[:,:,0] -= 103.939  # Blue 
        arr[:,:,1] -= 116.779  # Green 
        arr[:,:,2] -= 123.68   # Red 

        # Save as NumPy arrays
        save_path = os.path.join(processed_test_data, f"{os.path.splitext(f)[0]}.npy")
        np.save(save_path, arr)



