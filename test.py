import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.datasets import mnist
from keras.preprocessing import image

import picpro
from heatmap import *

# K.clear_session()
#
# model = VGG16(weights='imagenet')
# data_img = image.img_to_array(image.load_img('elephant.png'))
# # VGG16预处理:RGB转BGR,并对每一个颜色通道去均值中心化
# data_img = preprocess_input(data_img)
# img_show = image.img_to_array(image.load_img('elephant.png'))

# model = load_model('mnist2.h5')
# (x_train, y_train), (x_val, y_val) = mnist.load_data()
# x_train = np.expand_dims(x_train, axis=-1)
# x_val = np.expand_dims(x_val, axis=-1)
# data_img = x_train[17]

# model = load_model('train3.h5')
# x_train, y_train, x_val, y_val = picpro.csv2arr(r'test.csv', 10, (48,48,1))
# data_img = x_train[45]

# model = load_model('01.h5')
# x0_train, y0_train, x0_val, y0_val = picpro.csv2arr(r'0.csv', shape=(100, 100, 3))
# data_img = x0_train
#
# for i in range(20):
#     heatmaps(model, data_img[i])

#######################################################################################


