from keras import backend as K
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from keras.applications import VGG16

model = load_model('train3.h5')
# model = VGG16(weights='imagenet',
#               include_top=False)

# Ŀ�����
target = 11
# ��������
num_iterate = 100

# ----------------------------------���ӻ��˲���-------------------------------

# ������ͼ��ת������Чͼ��
def deprocess_image(x):
    # ���������й淶��
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    x += 0.5
    x = np.clip(x, 0, 1)
    # ת����RGB����
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x


# ͼ��ߴ��ͨ��
img_height, img_width, num_channels = K.int_shape(model.input)[1:4]
num_out = K.int_shape(model.layers[target].output)[-1]

for i_kernal in range(num_out):
    input_img = model.input
    # ����һ����ĺ�����ʹ�����ǵĲ�ĵ�n���˲����ļ�����󻯣�-1��softmax��
    # loss = K.mean(model.layers[-1].output[:, i_kernal])
    loss = K.mean(model.layers[target].output[:,:,:, i_kernal])  # m*28*28*128
    # ��������ͼ����ݶ��������ʧ
    grads = K.gradients(loss, input_img)[0]
    # Ч�ú���ͨ����L2������׼������
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    # �˺������ظ�������ͼ�����ĺ��ݶ�
    iterate = K.function([input_img], [loss, grads])
    # �Ӵ���һЩ��������Ļ�ɫͼ��ʼ
    np.random.seed(0)
    # ���ͼ��
    # input_img_data = np.random.randint(0, 255, (1, img_height, img_width, num_channels))  # ���
    # input_img_data = np.zeros((1, img_height, img_width, num_channels))   # ��ֵ
    input_img_data = np.random.random((1, img_height, img_width, num_channels)) * 20 + 128.   # ����Ҷ�
    input_img_data = np.array(input_img_data, dtype=float)
    failed = False
    # run gradient ascent
    print('####################################', i_kernal + 1)
    loss_value_pre = 0
    # �����ݶ�����100��
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
