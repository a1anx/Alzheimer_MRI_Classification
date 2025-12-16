# Alzheimer's Disease Classification from MRI Scans

E4040 Neural Networks and Deep Learning - Fall 2025 Project

## What Our Models Do

Classifies brain MRI scans into four Alzheimer's severity levels using transfer learning with CNNs. We trained Xception and InceptionResNetV2 models on ~33k augmented brain images and got 82%+ accuracy on the test set.

The models are particularly good at detecting severe cases (100% accuracy for ModerateDemented) but struggle with early-stage detection (65% F1 for VeryMildDemented).

## Dataset

Using the [Kaggle Multidisease Dataset](https://www.kaggle.com/datasets/praneshkumarm/multidiseasedataset) - specifically the Alzheimer's portion:
- 4 classes: NonDemented, VeryMildDemented, MildDemented, ModerateDemented
- Training: 4,712 images blown up to 32,984 after augmentation (7x)
- Validation: 671 images
- Test: 1,352 images

## Quick Start

### Setup
```bash
python -m venv venv
source venv/bin/activate  # Assuming mac
pip install -r requirements.txt
```

### Preprocessing
```bash
# After downloading data from Kaggle and unzip, which wil generate a folder called "Augumented Data": 

# 1. Split the dataset
python preprocessing/split_images.py

# 2. Preprocess for Xception/InceptionResNetV2 (299x299)
python preprocessing/xception_and_resnet/x_n_r_preprocess_images.py
```

### Training
```bash
# Train Xception
python model/xception/model_architecture/xception_train2.py

# Train InceptionResNetV2
python model/inceptionresnetv2/model_architecture/inceptionresnetv2_train.py
```

Each training run creates a timestamped folder in `model_output/` with:
- Trained model weights (.h5)
- Metrics and performance stats
- Training curves
- Confusion matrix

## Project Structure

```
├── model/
│   ├── xception/              # Xception model (82.17% test acc)
│   ├── inceptionresnetv2/     # InceptionResNetV2 (82.40% test acc)
│   ├── vgg16/                 # VGG16 (not trained)
│   └── vgg19/                 # VGG19 (not trained)
├── preprocessing/
│   ├── split_images.py        # Train/val/test split
│   ├── xception_and_resnet/   # 299x299 preprocessing
│   └── vgg16_and_vgg19/       # 224x224 preprocessing
├── Augumented Data/           # Original Kaggle dataset
├── References/                # Paper and reference notebooks
└── requirements.txt
```

## Results

| Model | Test Accuracy | Parameters | Training Time |
|-------|---------------|------------|---------------|
| InceptionResNetV2 | 82.40% | 57M (4.6% trainable) | 29 epochs |
| Xception | 82.17% | 24M (13.1% trainable) | 26 epochs |

Both models nail the severe cases but have trouble with early stage. Most errors happen between adjacent severity levels, which makes sense clinically.

## How It Works

1. Transfer Learning: We use ImageNet pre-trained models but freeze their weights. Only train the final classification layers (~3M parameters).

2. Data Augmentation: Rotation, shifting, zooming, and flipping to expand the training set 7x.

3. Memory Efficiency: Custom data generators load preprocessed .npy files sequentially instead of cramming everything into RAM.

4. Training Strategy: Adam optimizer, early stopping, learning rate reduction on plateau. Pretty standard stuff.

## Requirements

- Python 3.13
- TensorFlow 2.20
- Refer to `requirements.txt` for full dependencies

## Acknowledgements
The current project was run in Google Cloud Platform, with the following machine configurations:

Machine type : n1-standard-8 (8 vCPUs, 30 GB Memory)
CPU platform : Intel Broadwell
GPUs : 1 x NVIDIA T4

