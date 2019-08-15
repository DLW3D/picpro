import keras
from keras import layers
from keras import models

from keras.preprocessing.image import ImageDataGenerator

from picvisual import *

input_size = (150, 150)
batch_size = 256
# *************** Model *************** #
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

# *************** ImageDataGenerator *************** #
train_dir = r'cats_and_dogs_small/train'
validation_dir = r'cats_and_dogs_small/validation'

# All images will be rescaled by 1./255
# Data Augment
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,)
# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    # This is the target directory
    train_dir,
    # All images will be resized to 150x150
    target_size=input_size,
    batch_size=batch_size,
    # Since we use binary_crossentropy loss, we need binary labels
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=input_size,
    batch_size=batch_size,
    class_mode='binary')

# *************** Run *************** #
history = model.fit_generator(
      train_generator,
      steps_per_epoch=20,
      epochs=100,
      validation_data=validation_generator,
      validation_steps=10)

model.save('save.h5')















