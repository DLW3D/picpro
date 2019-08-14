import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.preprocessing import image


def heatmap(model, data_img, layer_idx, img_show=None, pred_idx=None):
    # 图像处理
    if data_img.shape.__len__() != 4:
        # 由于用作输入的img需要预处理,用作显示的img需要原图,因此分开两个输入
        if img_show is None:
            img_show = data_img
        # 缩放
        input_shape = K.int_shape(model.input)[1:3]     # (28,28)
        data_img = image.img_to_array(image.array_to_img(data_img).resize(input_shape))
        # 添加一个维度->(1, 224, 224, 3)
        data_img = np.expand_dims(data_img, axis=0)
    if pred_idx is None:
        # 预测
        preds = model.predict(data_img)
        # 获取最高预测项的index
        pred_idx = np.argmax(preds[0])
    # 目标输出估值
    target_output = model.output[:, pred_idx]
    # 目标层的输出代表各通道关注的位置
    last_conv_layer_output = model.layers[layer_idx].output
    # 求最终输出对目标层输出的导数(优化目标层输出),代表目标层输出对结果的影响
    grads = K.gradients(target_output, last_conv_layer_output)[0]
    # 将每个通道的导数取平均,值越高代表该通道影响越大
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer_output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([data_img])
    # 将各通道关注的位置和各通道的影响乘起来
    for i in range(conv_layer_output_value.shape[-1]):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    # 对各通道取平均得图片位置对结果的影响
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    # 规范化
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    # plt.matshow(heatmap)
    # plt.show()
    # 叠加图片
    # 缩放成同等大小
    heatmap = cv2.resize(heatmap, (img_show.shape[1], img_show.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    # 将热图应用于原始图像.由于opencv热度图为BGR,需要转RGB
    superimposed_img = img_show + cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)[:,:,::-1] * 0.4
    # 截取转uint8
    superimposed_img = np.minimum(superimposed_img, 255).astype('uint8')
    return superimposed_img, heatmap
    # 显示图片
    # plt.imshow(superimposed_img)
    # plt.show()
    # 保存为文件
    # superimposed_img = img + cv2.applyColorMap(heatmap, cv2.COLORMAP_JET) * 0.4
    # cv2.imwrite('ele.png', superimposed_img)

# 生成所有卷积层的热度图
def heatmaps(model, data_img, img_show=None):
    if img_show is None:
        img_show = np.array(data_img)
    # Resize
    input_shape = K.int_shape(model.input)[1:3]  # (28,28,1)
    data_img = image.img_to_array(image.array_to_img(data_img).resize(input_shape))
    # 添加一个维度->(1, 224, 224, 3)
    data_img = np.expand_dims(data_img, axis=0)
    # 预测
    preds = model.predict(data_img)
    # 获取最高预测项的index
    pred_idx = np.argmax(preds[0])
    print("预测为:%d(%f)" % (pred_idx, preds[0][pred_idx]))
    indexs = []
    for i in range(model.layers.__len__()):
        if 'conv' in model.layers[i].name:
            indexs.append(i)
    print('模型共有%d个卷积层' % indexs.__len__())
    plt.suptitle('heatmaps for each conv')
    for i in range(indexs.__len__()):
        ret = heatmap(model, data_img, indexs[i], img_show=img_show, pred_idx=pred_idx)
        plt.subplot(np.ceil(np.sqrt(indexs.__len__()*2)), np.ceil(np.sqrt(indexs.__len__()*2)), i*2 + 1)\
            .set_title(model.layers[indexs[i]].name)
        plt.imshow(ret[0])
        plt.axis('off')
        plt.subplot(np.ceil(np.sqrt(indexs.__len__()*2)), np.ceil(np.sqrt(indexs.__len__()*2)), i*2 + 2)\
            .set_title(model.layers[indexs[i]].name)
        plt.imshow(ret[1])
        plt.axis('off')
    plt.show()














