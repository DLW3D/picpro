from keras import backend as K
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

model = load_model('train3.h5')
# 图像通道
num_channels = 1
# 输入图像尺寸
img_height = img_width = 48
# 输出标签数
num_label = 7

# ----------------------------------可视化滤波器-------------------------------

# 将张量转换成有效图像
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


for i_kernal in range(num_label):
    input_img = model.input
    # 构建一个损耗函数，使所考虑的层的第n个滤波器的激活最大化，-1层softmax层
    loss = K.mean(model.layers[-1].output[:, i_kernal])
    # loss = K.mean(model.output[:, :,:, i_kernal])
    # 计算输入图像的梯度与这个损失
    grads = K.gradients(loss, input_img)[0]
    # 效用函数通过其L2范数标准化张量
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    # 此函数返回给定输入图像的损耗和梯度
    iterate = K.function([input_img], [loss, grads])
    # 从带有一些随机噪声的灰色图像开始
    np.random.seed(0)
    # 随机图像
    input_img_data = np.random.randint(0, 255, (1, img_height, img_width, num_channels))
    input_img_data = np.array(input_img_data, dtype=float)
    failed = False
    # run gradient ascent
    print('####################################', i_kernal + 1)
    loss_value_pre = 0
    # 运行梯度上升500步
    for i in range(500):
        loss_value, grads_value = iterate([input_img_data])
        if i % 100 == 0:
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
        input_img_data += grads_value * 255 * 1  # e-3
    plt.subplot(2, 5, i_kernal + 1)
    # plt.imshow((process(input_img_data[0,:,:,0])*255).astype('uint8'), cmap='Greys') #cmap='Greys'
    # img_re = deprocess_image(input_img_data[0])
    img_re = input_img_data[0].astype(np.uint8)
    if num_channels == 1:
        img_re = np.reshape(img_re, (img_height, img_width))
    else:
        img_re = np.reshape(img_re, (img_height, img_width, num_channels))
    plt.imshow(img_re)  # , cmap='gray'

plt.show()
