from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import tensorflow as tf 
import inceptionresnetv2_config


class InceptionResNetV2_Finetune_Model():
    def __init__(
        self,
        dense1_size: int,
        dense2_size: int, 
        dropout: float = None, 
    ):
        super().__init__()

        # Use config values 
        self.dense1_size = con



base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(299,299,3))
x = base_model.outpput
x = GlobalAveragePooling2D()(x)
x = tf.keras.layers.Flatten()
x = Dense(2048, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(, epochs=, validation_data=)
