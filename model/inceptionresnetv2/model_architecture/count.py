import tensorflow as tf
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten
from tensorflow.keras.models import Model

# Build the model
base = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(299,299,3))
base.trainable = False


x = base.output
x = GlobalAveragePooling2D()(x)
x = Flatten()(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(4, activation='softmax')(x)

model = Model(inputs=base.input, outputs=output)

# Count parameters
trainable_params = sum([tf.size(var).numpy() for var in model.trainable_variables])
non_trainable_params = sum([tf.size(var).numpy() for var in model.non_trainable_variables])
total_params = trainable_params + non_trainable_params

print(f"\nTrainable parameters: {trainable_params:,}")
print(f"Non-trainable parameters: {non_trainable_params:,}")
print(f"Total parameters: {total_params:, }")

# Show detailed layer info
print("\nLayer breakdown:")
model.summary()

# Display input and output shapes
print(f"\nModel Input shape: {model.input_shape}")
print(f"Model Output Shape: {model.output_shape}")