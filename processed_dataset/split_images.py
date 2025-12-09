"""
After downloading and unzipping Multidisease images from:
https://www.kaggle.com/datasets/praneshkumarm/multidiseasedataset/code

Run this script ONCE to:
1) Split images to 0.7 (TRAIN), 0.1 (VAL), 0.2 (TEST)
2) Remove any non-Alzheimer images
"""

import pandas as pd
import numpy as np 
import argparse 
from typing import List, Set, Tuple
import json
import splitfolders
import directories as dir
import os
import shutil

# Split images into designated ratios
splitfolders.ratio(dir.INIT_INPUT, output=dir.INIT_OUTPUT, seed=123, ratio=(0.7, 0.1, 0.2), group_prefix=None)

# Delete non-Alzheimer images in train set
for filename in os.listdir(dir.TRAIN):
    if filename.startswith("."):
        continue 
    path = os.path.join(dir.TRAIN, filename)
    if "Alzheimer" not in filename:
        shutil.rmtree(path)

# Delete non-Alzheimer images in val set
for filename in os.listdir(dir.VAL):
    if filename.startswith("."):
        continue 
    path = os.path.join(dir.VAL, filename)
    if "Alzheimer" not in filename:
        shutil.rmtree(path)

# Delete non-Alzheimer images in test set
for filename in os.listdir(dir.TEST):
    if filename.startswith("."):
        continue 
    path = os.path.join(dir.TEST, filename)
    if "Alzheimer" not in filename:
        shutil.rmtree(path)
        print(f"Deleted: {filename}")

# Count train images for confirmation
TRAIN_total = 0
for root, dirs, files in os.walk(dir.TRAIN):
    for f in files:
        if f.lower().endswith(".jpeg"):
            TRAIN_total += 1

# Count val images for confirmation
VAL_total = 0
for root, dirs, files in os.walk(dir.VAL):
    for f in files:
        if f.lower().endswith(".jpeg"):
            VAL_total += 1

# Count test images for confirmation
TEST_total = 0
for root, dirs, files in os.walk(dir.TEST):
    for f in files:
        if f.lower().endswith(".jpeg"):
            TEST_total += 1

print("Train:", TRAIN_total)
print("Val:", VAL_total)
print("Test:", TEST_total)




