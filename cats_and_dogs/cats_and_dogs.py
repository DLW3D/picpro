import keras
from keras import layers
from keras import models
from keras.applications import VGG16

from keras.preprocessing.image import ImageDataGenerator

input_size = (150, 150)
batch_size = 256
# *************** Model *************** #
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu',
#                         input_shape=(150, 150, 3)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Flatten())
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(512, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))
#
# model.compile(loss=keras.losses.binary_crossentropy,
#               optimizer=keras.optimizers.RMSprop(lr=1e-4),
#               metrics=['acc'])
#
# model.summary()

# *************** Model use pre-train *************** #
conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# 冻结部分网络
model.layers[0].trainable = False

model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

model.summary()

# *************** ImageDataGenerator *************** #
train_dir = r'cats_and_dogs_small/train'
validation_dir = r'cats_and_dogs_small/validation'

# 训练集数据增强
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,)

# 测试集
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    # 目标文件夹
    train_dir,
    # 规范化图片大小
    target_size=input_size,
    batch_size=batch_size,
    # 二分类
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=input_size,
    batch_size=batch_size,
    class_mode='binary')

# *************** Run *************** #
history = model.fit_generator(
      train_generator,
      steps_per_epoch=10,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=10)

model.save('save.h5')















