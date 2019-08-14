# !/usr/bin/python
# -*- coding:utf8 -*-
import itertools

import keras
import matplotlib.pyplot as plt
from keras.engine.saving import load_model
from sklearn.metrics import confusion_matrix

from picpro import *


# 模型保存与读取
def save(model, path):
    model.save(path)


def load(path):
    return load_model(path)


# 使用模型判断图片*********
def tester(model, path):
    tester = np.array(scale(arr2img(read_img2arr(path)), 100, 100)).reshape(1, 100, 100, 3)
    return model.predict(tester)

# 检查测试数据分布情况
def check_label(y_train):
    count = np.zeros(y_train.shape[1], dtype=int)
    for i in y_train.argmax(axis=-1):
        count[i] += 1
    return count


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
    truelabel = y_val.argmax(axis=-1)  # 将one-hot转化为label
    conf_mat = confusion_matrix(y_true=truelabel, y_pred=predictions)
    plt.figure()
    plot_confusion_matrix(conf_mat, range(np.max(truelabel)+1))


# 卷积网络可视化
def visual(model, data, num_layer=1):
    # data:array数据
    # layer:第n层的输出
    data = np.expand_dims(data, axis=0)  # 开头加一维
    layer = keras.backend.function([model.layers[0].input], [model.layers[num_layer].output])
    f1 = layer([data])[0]
    num = f1.shape[-1]
    plt.figure(figsize=(8, 8))
    for i in range(num):
        plt.subplot(np.ceil(np.sqrt(num)), np.ceil(np.sqrt(num)), i + 1)
        plt.imshow(f1[0, :, :, i] * 255)  # , cmap='gray'
        plt.axis('off')
    plt.show()
