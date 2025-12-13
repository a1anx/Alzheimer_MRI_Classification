from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import tensorflow as tf 
import sys
import os

# Add parent directory path to import directories module 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

class xception():
    def __init__(
        self,
        dense1_size: int,
        dense2_size: int, 
        dropout: float = None, 
    ):
        # super().__init__()

        # Use config values if not specified
        # self.dense1_size = dense1_size or config.MODEL['dense1_size']
        # self.dense2_size = dense2_size or config.MODEL['dense2_size']
        self.dropout = dropout or config.MODEL['dropout']

        base = Xception(
            weights='imagenet', 
            include_top=False, 
            input_shape=(299,299,3)
        )
        #Transfer learning needs to freeze pre-trained model
        base.trainable = False
        x = base.output
        #Based on paper, we add the following layers
        #Global Average Pooling 2D. This also covers the flattening layer
        x = GlobalAveragePooling2D()(x)
        #Dropout 1
        x = Dropout(self.dropout)(x)
        #Dense 1
        x = Dense(1024, activation='relu')(x)
        #Dense 2
        x = Dense(1024, activation='relu')(x)
        #Dropout 2
        x = Dropout(self.dropout)(x)
        #Dense 3
        output = Dense(4, activation='softmax')(x)

        self.model = Model(inputs=base.input, outputs=output)