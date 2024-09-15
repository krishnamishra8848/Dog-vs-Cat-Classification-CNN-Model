import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Updated paths
train_dir = '/content/training_set/training_set'
test_dir = '/content/test_set/test_set'

# Image preprocessing and augmentation
img_size = (128, 128)  # Resize all images to 128x128
batch_size = 32

# Apply data augmentation and rescaling
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

# Define data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.regularizers import l2

def build_model_with_l2():
    model = Sequential()

    # Conv layers with He normal initialization, padding same, and L2 regularization
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu',
                     kernel_initializer=HeNormal(), kernel_regularizer=l2(0.001), input_shape=(128, 128, 3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu',
                     kernel_initializer=HeNormal(), kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu',
                     kernel_initializer=HeNormal(), kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    
    model.add(Dense(512, activation='relu', kernel_initializer=HeNormal(), kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(0.001)))  # Binary classification

    return model


# Compile model with Adam optimizer and binary cross-entropy loss
model = build_model()
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Learning rate scheduler to reduce LR after specific epochs
def lr_scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

lr_callback = LearningRateScheduler(lr_scheduler)

# Early stopping and checkpointing
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

# Train model with early stopping, checkpoint, and LR scheduler
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=test_generator,
    callbacks=[early_stopping, checkpoint, lr_callback]
)
# Load the best saved model
best_model = tf.keras.models.load_model('best_model.h5')

# Evaluate the model on the test set
loss, accuracy = best_model.evaluate(test_generator)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')

Epoch 1/20
251/251 [==============================] - 64s 248ms/step - loss: 0.7787 - accuracy: 0.6450 - val_loss: 0.7249 - val_accuracy: 0.6307 - lr: 0.0010
Epoch 2/20
251/251 [==============================] - 61s 242ms/step - loss: 0.5694 - accuracy: 0.7232 - val_loss: 0.5326 - val_accuracy: 0.7415 - lr: 0.0010
Epoch 3/20
251/251 [==============================] - 60s 240ms/step - loss: 0.5163 - accuracy: 0.7490 - val_loss: 0.5649 - val_accuracy: 0.7484 - lr: 0.0010
Epoch 4/20
251/251 [==============================] - 60s 240ms/step - loss: 0.4971 - accuracy: 0.7592 - val_loss: 0.5102 - val_accuracy: 0.7682 - lr: 0.0010
Epoch 5/20
251/251 [==============================] - 60s 240ms/step - loss: 0.4423 - accuracy: 0.7931 - val_loss: 0.4347 - val_accuracy: 0.8092 - lr: 0.0010
Epoch 6/20
251/251 [==============================] - 60s 238ms/step - loss: 0.4219 - accuracy: 0.8066 - val_loss: 0.5247 - val_accuracy: 0.7588 - lr: 0.0010
Epoch 7/20
251/251 [==============================] - 60s 237ms/step - loss: 0.4011 - accuracy: 0.8214 - val_loss: 0.5716 - val_accuracy: 0.7514 - lr: 0.0010
Epoch 8/20
251/251 [==============================] - 60s 240ms/step - loss: 0.3729 - accuracy: 0.8312 - val_loss: 0.3908 - val_accuracy: 0.8379 - lr: 0.0010
Epoch 9/20
251/251 [==============================] - 60s 237ms/step - loss: 0.3566 - accuracy: 0.8437 - val_loss: 0.5126 - val_accuracy: 0.7929 - lr: 0.0010
Epoch 10/20
251/251 [==============================] - 60s 238ms/step - loss: 0.3326 - accuracy: 0.8551 - val_loss: 0.4073 - val_accuracy: 0.8374 - lr: 0.0010
Epoch 11/20
251/251 [==============================] - 60s 237ms/step - loss: 0.3090 - accuracy: 0.8653 - val_loss: 0.4249 - val_accuracy: 0.8146 - lr: 9.0484e-04
Epoch 12/20
251/251 [==============================] - 60s 240ms/step - loss: 0.2688 - accuracy: 0.8867 - val_loss: 0.3762 - val_accuracy: 0.8478 - lr: 8.1873e-04
Epoch 13/20
251/251 [==============================] - 60s 237ms/step - loss: 0.2678 - accuracy: 0.8884 - val_loss: 0.9010 - val_accuracy: 0.6846 - lr: 7.4082e-04
Epoch 14/20
251/251 [==============================] - 60s 238ms/step - loss: 0.2637 - accuracy: 0.8899 - val_loss: 0.3907 - val_accuracy: 0.8468 - lr: 6.7032e-04
Epoch 15/20
251/251 [==============================] - 60s 240ms/step - loss: 0.2263 - accuracy: 0.9072 - val_loss: 0.3624 - val_accuracy: 0.8473 - lr: 6.0653e-04
Epoch 16/20
251/251 [==============================] - 60s 240ms/step - loss: 0.2035 - accuracy: 0.9172 - val_loss: 0.3541 - val_accuracy: 0.8606 - lr: 5.4881e-04
Epoch 17/20
251/251 [==============================] - 60s 237ms/step - loss: 0.1895 - accuracy: 0.9230 - val_loss: 0.5747 - val_accuracy: 0.8047 - lr: 4.9659e-04
Epoch 18/20
251/251 [==============================] - 61s 241ms/step - loss: 0.1835 - accuracy: 0.9272 - val_loss: 0.3229 - val_accuracy: 0.8665 - lr: 4.4933e-04
Epoch 19/20
251/251 [==============================] - 60s 239ms/step - loss: 0.1738 - accuracy: 0.9304 - val_loss: 0.3217 - val_accuracy: 0.8898 - lr: 4.0657e-04
Epoch 20/20
251/251 [==============================] - 59s 236ms/step - loss: 0.1479 - accuracy: 0.9412 - val_loss: 0.3461 - val_accuracy: 0.8705 - lr: 3.6788e-04
64/64 [==============================] - 4s 61ms/step - loss: 0.3217 - accuracy: 0.8898
Test Loss: 0.32168304920196533
Test Accuracy: 0.8897676467895508

