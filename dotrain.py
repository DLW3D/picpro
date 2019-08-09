# !/usr/bin/python
# -*- coding:utf8 -*-
import random

import numpy as np
import keras
from keras.engine.saving import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import backend as K, callbacks
from keras_preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

import picpro


# 屏蔽WARNING
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


batch_size = 100
num_classes = 10
epochs = 12

input_shape = (28,28,1)

# 加载数据集
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 格式化数据集
x_train = x_train.reshape(60000,28,28,1) # 将图片摊平，变成向量
x_test = x_test.reshape(10000,28,28,1) # 对测试集进行同样的处理
y_train = keras.utils.to_categorical(y_train,10)# 将标签展开
y_test = keras.utils.to_categorical(y_test,10)
# 归一化数据
x_train = x_train / 255
x_test = x_test / 255


# input_shape = (100, 100, 3)
#
# x0_train, y0_train, x0_test, y0_test = picpro.ReadFile(r'0.csv', input_shape)
# x1_train, y1_train, x1_test, y1_test = picpro.ReadFile(r'1.csv', input_shape)
#
# print('正在处理训练集...')
# # 合成训练集
# x_train = np.concatenate([x1_train, x0_train], 0)
# y_train = np.concatenate([y1_train, y0_train], 0)
# x_test = np.concatenate([x1_test, x0_test], 0)
# y_test = np.concatenate([y1_test, y0_test], 0)
# del x0_train, y0_train, x0_test, y0_test
# del x1_train, y1_train, x1_test, y1_test
# # 打乱训练集
# index = [i for i in range(len(y_train))]
# random.shuffle(index)
# x_train = x_train[index]
# y_train = y_train[index]
# # 添加噪声
# # for i in x_train:
# #     i += np.random.rand(input_shape)/25
# # for i in x_test:
# #     i += np.random.rand(input_shape)/25



print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# 测试
# picpro.ArrayToImage(x_train[0]*255).show()

# convert class vectors to binary class matrices 将类向量转换为二进制类矩阵(one-hot)
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)

print("正在构建模型...")
# build the neural net 建模型(卷积—relu-卷积-relu-池化-relu-卷积-relu-池化-全连接)
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))  # 32个过滤器，过滤器大小是3×3，32×26×26
model.add(Conv2D(64, (3, 3), activation='relu'))  # 64×24×24
model.add(MaxPooling2D(pool_size=(2, 2)))  # 向下取样
model.add(Dropout(0.25))
model.add(Flatten())  # 降维：将64×12×12降为1维（即把他们相乘起来）
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))  # 全连接2层
# 显示模型
model.summary()

# compile the model 编译模型
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# Call Back
# 运行 tensorboard --logdir="C:\Users\78753\Desktop\DL\picpro\logs" 访问 http://localhost:6006/ 查看
tb_config = keras.callbacks.TensorBoard(
    log_dir=r'.\logs',
    write_graph=True,
    write_images=True,
    histogram_freq=1)
cbks = [callbacks.EarlyStopping(patience=0, mode='auto', verbose=1), tb_config]

# train the model 训练模型
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    callbacks=cbks,
                    validation_data=(x_test, y_test))


# # 使用Generator进行图像增强(增加数据集大小)
# datagen_train = ImageDataGenerator(rotation_range=3.)
# datagen_train.fit(x_train)
#
# # 使用Generator产生的数据进行训练
# history = model.fit_generator(datagen_train.flow(x_train, y_train, batch_size=batch_size),
#                               steps_per_epoch=500,
#                               epochs=epochs,
#                               verbose=1)

# test the model 测试模型
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# 模型保存与读取
def Save(path):
    model.save(path)


def Load(path):
    return load_model(path)


# 可视化
def plot_train_history(history, train_metrics):
    plt.plot(history.history.get(train_metrics))
    plt.ylabel(train_metrics)
    plt.xlabel('Epochs')
    plt.legend(['train'])


def plot(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plot_train_history(history, 'loss')
    plt.subplot(1, 2, 2)
    plot_train_history(history, 'acc')
    plt.show()
