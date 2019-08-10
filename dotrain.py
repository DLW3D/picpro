# !/usr/bin/python
# -*- coding:utf8 -*-
import itertools
import random

import numpy as np
import keras
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, CSVLogger
from keras.datasets import mnist
from keras.engine.saving import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, LeakyReLU, ReLU
from keras import callbacks
from keras_preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import picpro

batch_size = 256
num_classes = 7
epochs = 100

input_shape = (48, 48, 1)

# x0_train, y0_train, x0_val, y0_val = picpro.ReadFile(r'0.csv', input_shape)
# x1_train, y1_train, x1_val, y1_val = picpro.ReadFile(r'1.csv', input_shape)


print('正在处理训练集...')
# 合成训练集
# x_train = np.concatenate([x1_train, x0_train], 0)
# y_train = np.concatenate([y1_train, y0_train], 0)
# x_val = np.concatenate([x1_val, x0_val], 0)
# y_val = np.concatenate([y1_val, y0_val], 0)
# del x0_train, y0_train, x0_val, y0_val
# del x1_train, y1_train, x1_val, y1_val

x_train, y_train, x_val, y_val = picpro.ReadFile(r'train.csv', 10, input_shape)

# 打乱训练集
# index = [i for i in range(len(y_train))]
# random.shuffle(index)
# x_train = x_train[index]
# y_train = y_train[index]
# 添加噪声
# for i in x_train:
#     i += np.random.rand(input_shape)/25
# for i in x_val:
#     i += np.random.rand(input_shape)/25
# 将类向量转换为二进制类矩阵(one-hot)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)

print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print(x_train.shape[0], 'train samples')
print(x_val.shape[0], 'val samples')

# 测试
# picpro.ArrayToImage(x_train[0]).show()


print("正在构建模型...")
# 设置 Call Back
# TensorBoard 运行 tensorboard --logdir="C:\Users\78753\Desktop\DL\picpro\logs" 访问 http://localhost:6006/ 查看
tb_config = keras.callbacks.TensorBoard(
    log_dir=r'.\logs',
    write_graph=True,
    write_images=True,
    histogram_freq=1)
early_stopping = callbacks.EarlyStopping(patience=20, mode='auto', verbose=1)
lrate = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1, mode='auto', epsilon=0.0001, cooldown=0,
                          min_lr=0.00001)
checkpoint = ModelCheckpoint(filepath='save.h5', monitor='val_acc', verbose=1, save_best_only=True,
                             save_weights_only=False, mode='auto', period=1)
earlystopping = EarlyStopping(monitor='val_acc', patience=20, verbose=1, mode='auto')
csvlogger = CSVLogger('log_save.csv', append=False)
cbks = [lrate, earlystopping, tb_config]  # , checkpoint, csvlogger]


# 构建模型
myseed = 10
drop_rate = 0.25
model = Sequential()
model.add(BatchNormalization(input_shape=input_shape))  # 批量规范化层(不用手动规范化数据了)
model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal'))   # 卷积
model.add(LeakyReLU())  # 激活
model.add(BatchNormalization())     # 批量规范化层
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))   # 池化
model.add(Dropout(rate=drop_rate, seed=myseed))     # Dropout

model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal'))
model.add(LeakyReLU())
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(rate=drop_rate, seed=myseed))

model.add(Conv2D(filters=256, kernel_size=(6, 6), padding='same', kernel_initializer='glorot_normal'))
model.add(LeakyReLU())
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(rate=drop_rate, seed=myseed))

model.add(Conv2D(filters=256, kernel_size=(6, 6), padding='same', kernel_initializer='glorot_normal'))
model.add(LeakyReLU())
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(rate=drop_rate, seed=myseed))

model.add(Flatten())

model.add(Dense(256, kernel_initializer='glorot_normal'))
model.add(LeakyReLU())
model.add(BatchNormalization())
model.add(Dropout(rate=drop_rate, seed=myseed))

model.add(Dense(256, kernel_initializer='glorot_normal'))
model.add(LeakyReLU())
model.add(BatchNormalization())
model.add(Dropout(rate=drop_rate, seed=myseed))

model.add(Dense(units=7, activation='softmax', kernel_initializer='glorot_normal'))
# compile the model 编译模型
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.adamax(),
              metrics=['accuracy'])
# 显示模型
model.summary()


# 使用Generator进行图像增强(增加数据集大小)
generate = ImageDataGenerator(rotation_range=30, width_shift_range=0.2, height_shift_range=0.2, zoom_range=[0.8, 1.2],
                              shear_range=0.2, horizontal_flip=True)
generate.fit(x_train)
history = model.fit_generator(generate.flow(x_train, y_train, batch_size=batch_size),
                              validation_data=(x_val, y_val),
                              steps_per_epoch=int(len(x_train) / batch_size),
                              epochs=epochs,
                              verbose=1,
                              callbacks=cbks)

# train the model 训练模型
# history = model.fit(x_train, y_train,
#                     batch_size=batch_size,
#                     epochs=epochs,
#                     verbose=1,
#                     callbacks=cbks,
#                     validation_data=(x_val, y_val))


# val the model 测试模型
score = model.evaluate(x_val, y_val, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# 模型保存与读取
def Save(path):
    model.save(path)


def Load(path):
    return load_model(path)


# 可视化
def show_train_history(train_history, train_metrics, validation_metrics):
    plt.plot(train_history.history[train_metrics])
    plt.plot(train_history.history[validation_metrics])
    plt.title('Train History')
    plt.ylabel(train_metrics)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')

# 显示训练过程
def plot(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    show_train_history(history, 'acc', 'val_acc')
    plt.subplot(1, 2, 2)
    show_train_history(history, 'loss', 'val_loss')
    plt.show()


def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.jet):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# 显示混淆矩阵
def plot_confuse(model, x_val, y_val):
    predictions = model.predict_classes(x_val)
    truelabel = y_val.argmax(axis=-1)   # 将one-hot转化为label
    conf_mat = confusion_matrix(y_true=truelabel, y_pred=predictions)
    plt.figure()
    plot_confusion_matrix(conf_mat, [])


# 卷积网络可视化
def visual(data, num_layer=1):
    # data:array数据
    # layer:第n层的输出
    data = np.expand_dims(data, axis=0)     # 开头加一维
    layer = keras.backend.function([model.layers[0].input], [model.layers[num_layer].output])
    f1 = layer([data])[0]
    # picpro.ArrayToImage(f1[0, :, :, 0] * 255).show()
    num = f1.shape[-1]
    plt.figure(figsize=(8, 8))
    for i in range(num):
        plt.subplot(np.ceil(np.sqrt(num)), np.ceil(np.sqrt(num)), i+1)
        plt.imshow(f1[0, :, :, i] * 255, cmap='gray')
        plt.axis('off')
    plt.show()


