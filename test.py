import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image

K.clear_session()

# Note that we are including the densely-connected classifier on top;
# all previous times, we were discarding it.
model = VGG16(weights='imagenet')

# The local path to our target image
img_path = 'elephant.png'
# `img` is a PIL image of size 224x224
img = image.load_img(img_path, target_size=(224, 224))
# `x` is a float32 Numpy array of shape (224, 224, 3)
x = image.img_to_array(img)
# 添加一个维度->(1, 224, 224, 3)
x = np.expand_dims(x, axis=0)
# 去均值中心化(平均值取0)
x = preprocess_input(x)

# 预测
preds = model.predict(x)
# 获取最高预测项的index
index = np.argmax(preds[0])
# 这是预测向量中的“非洲象”条目
african_elephant_output = model.output[:, 386]

# 获取最后一个卷积层
last_conv_layer = model.get_layer('block5_conv3')

# This is the gradient of the "african elephant" class with regard to
# the output feature map of `block5_conv3`
# 求最终输出对目标层输出的导数(优化目标层输出?)
grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]

# This is a vector of shape (512,), where each entry
# is the mean intensity of the gradient over a specific feature map channel
pooled_grads = K.mean(grads, axis=(0, 1, 2))

# This function allows us to access the values of the quantities we just defined:
# `pooled_grads` and the output feature map of `block5_conv3`,
# given a sample image
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

# These are the values of these two quantities, as Numpy arrays,
# given our sample image of two elephants
pooled_grads_value, conv_layer_output_value = iterate([x])

# We multiply each channel in the feature map array
# by "how important this channel is" with regard to the elephant class
for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

# 对各通道取平均
heatmap = np.mean(conv_layer_output_value, axis=-1)

# 规范化
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
# plt.matshow(heatmap)
# plt.show()

# 叠加图片
img = cv2.imread(img_path)
# 缩放成同等大小
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)

# 将热图应用于原始图像
superimposed_img = img + cv2.applyColorMap(255-heatmap, cv2.COLORMAP_JET) * 0.4
superimposed_img = (superimposed_img * 255 / np.max(superimposed_img)).astype('uint8')
plt.imshow(superimposed_img)
plt.show()

# 保存为文件
# superimposed_img = img + cv2.applyColorMap(heatmap, cv2.COLORMAP_JET) * 0.4
# cv2.imwrite('/Users/fchollet/Downloads/elephant_cam.jpg', superimposed_img)
