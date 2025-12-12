from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import tensorflow as tf 
import sys
import os

# Add parent directory path to import directories module 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

class inceptionresnetv2():
    def __init__(
        self,
        dense1_size: int,
        dense2_size: int, 
        dropout: float = None, 
    ):
        super().__init__()

        # Use config values if not specified
        self.dense1_size = dense1_size or config.MODEL['dense1_size']
        self.dense2_size = dense2_size or config.MODEL['dense2_size']
        self.dropout = dropout or config.MODEL['dropout']

        base = InceptionResNetV2(
            weights='imagenet', 
            include_top=False, 
            input_shape=(299,299,3)
        )
        x = base.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(self.dense1_size, activation='relu')(x)
        x = Dropout(self.dropout)(x)
        x = Dense(self.dense2_size, activation='relu')(x)
        x = Dropout(self.dropout)(x)
        output = Dense(4, activation='softmax')(x)

        self.model = Model(inputs=base.input, outputs=output)


