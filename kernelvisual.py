import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.models import load_model

# 加载模型
model = load_model('mnist2.h5')
# 目标层数
target = 11
# 迭代次数
num_iterate = 100

# ----------------------------------可视化滤波器-------------------------------

# 将浮点图像转换成有效图像
def deprocess_image(x):
    # 对张量进行规范化
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    x += 0.5
    x = np.clip(x, 0, 1)
    # 转化到RGB数组
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x


# 图像尺寸和通道
img_height, img_width, num_channels = K.int_shape(model.input)[1:4]
num_out = K.int_shape(model.layers[target].output)[-1]

for i_kernal in range(num_out):
    input_img = model.input
    # 构建一个损耗函数，使所考虑的层的第n个滤波器的激活最大化，-1层softmax层
    # loss = K.mean(model.layers[-1].output[:, i_kernal])
    loss = K.mean(model.layers[target].output[:,:,:, i_kernal])  # m*28*28*128
    # 计算图像对损失函数的梯度
    grads = K.gradients(loss, input_img)[0]
    # 效用函数通过其L2范数标准化张量
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    # 此函数返回给定输入图像的损耗和梯度
    iterate = K.function([input_img], [loss, grads])
    # 从带有一些随机噪声的灰色图像开始
    np.random.seed(0)
    # 随机图像
    # input_img_data = np.random.randint(0, 255, (1, img_height, img_width, num_channels))  # 随机
    # input_img_data = np.zeros((1, img_height, img_width, num_channels))   # 零值
    input_img_data = np.random.random((1, img_height, img_width, num_channels)) * 20 + 128.   # 随机灰度
    input_img_data = np.array(input_img_data, dtype=float)
    failed = False
    # run gradient ascent
    print('####################################', i_kernal + 1)
    loss_value_pre = 0
    # 运行梯度上升100步
    for i in range(num_iterate):
        loss_value, grads_value = iterate([input_img_data])
        if i % 20 == 0:
            # print(' predictions: ' , np.shape(predictions), np.argmax(predictions))
            print('Iteration %d/%d, loss: %f' % (i, 500, loss_value))
            print('Mean grad: %f' % np.mean(grads_value))
            if all(np.abs(grads_val) < 0.000001 for grads_val in grads_value.flatten()):
                failed = True
                print('Failed')
                break
            # print('Image:\n%s' % str(input_img_data[0,0,:,:]))
            if loss_value_pre != 0 and loss_value_pre > loss_value:
                break
            if loss_value_pre == 0:
                loss_value_pre = loss_value
            # if loss_value > 0.99:
            #     break
        input_img_data += grads_value * 1  # e-3
    img_re = deprocess_image(input_img_data[0])
    if num_channels == 1:
        img_re = np.reshape(img_re, (img_height, img_width))
    else:
        img_re = np.reshape(img_re, (img_height, img_width, num_channels))
    plt.subplot(np.ceil(np.sqrt(num_out)), np.ceil(np.sqrt(num_out)), i_kernal + 1)
    plt.imshow(img_re)  # , cmap='gray'

plt.show()
