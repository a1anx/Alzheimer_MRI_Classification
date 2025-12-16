from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten
from tensorflow.keras.models import Model
import tensorflow as tf 
import sys
import os
from pathlib import Path

# Add parent directory path to import directories module 
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

class inceptionresnetv2():
    def __init__(
        self,
        dense1_size: int,
        dense2_size: int, 
        dropout: float = None, 
    ):
        # Use config values if not specified
        self.dense1_size = dense1_size or config.MODEL['dense1_size']
        self.dense2_size = dense2_size or config.MODEL['dense2_size']
        self.dropout = dropout or config.MODEL['dropout']

        # Import InceptionResnetV2
        base = InceptionResNetV2(
            weights='imagenet', 
            include_top=False, 
            input_shape=(299,299,3)
        )
        base.trainable = False
        x = base.output
        x = GlobalAveragePooling2D()(x)
        x = Flatten()(x)
        x = Dense(self.dense2_size, activation='relu')(x)
        x = Dropout(self.dropout)(x)
        x = Dense(self.dense2_size, activation='relu')(x)
        x = Dropout(self.dropout)(x)
        output = Dense(4, activation='softmax')(x)

        self.model = Model(inputs=base.input, outputs=output)


