"""
Pathnames for input and output for preprocessing
Make sure to create a folder called "split_augmented_data"
"""
import os

# Pathname for project root (automatically detects based on this file's location)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Pathname for initial Augmented Data folder (before running splitfolders)
INIT_INPUT = os.path.join(ROOT, 'Augumented Data')

# Pathname for MRI scan images in train, validation, and test sets in  (before running preprocessing)
TEST = os.path.join(ROOT, 'split_augmented_data', 'test')
VAL = os.path.join(ROOT, 'split_augmented_data', 'val')
TRAIN = os.path.join(ROOT, 'split_augmented_data', 'train')

# Pathname for processed_data per model (for running preprocessing)
X_N_R = os.path.join(ROOT, 'preprocessing', 'xception_and_resnet')
V_N_V = os.path.join(ROOT, 'preprocessing', 'vgg16_and_vgg19')

 








 