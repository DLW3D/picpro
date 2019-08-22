import msvcrt
import os
import time

import tensorflow as tf
from keras import backend as K
import keras.backend.tensorflow_backend as KTF
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import keras
from keras import layers, Sequential, models
import numpy as np

# 手动分配GPU
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
# config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 指定分配50%空间
sess = tf.Session(config=config)  # 设置session
KTF.set_session(sess)

# IO参数
latent_dim = 100
img_shape = (96, 96, 3)


# ************************** 生成器
def build_generator():
    model = Sequential()
    model.add(layers.Dense(512 * 6 * 6, activation='relu', input_dim=latent_dim))  # 输入维度为100
    model.add(layers.Reshape((6, 6, 512)))
    model.add(layers.Conv2DTranspose(256, 5, strides=2, padding='same'))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2DTranspose(128, 5, strides=2, padding='same'))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2DTranspose(64, 5, strides=2, padding='same'))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2DTranspose(img_shape[-1], 5, strides=2, padding='same'))
    model.add(layers.Activation("tanh"))
    model.summary()  # 打印网络参数
    noise = models.Input(shape=(latent_dim,))
    img = model(noise)
    return models.Model(noise, img)  # 定义一个 一个输入noise一个输出img的模型


# ************************** 判别器
def build_discriminator():
    dropout = 0.4
    model = Sequential()
    model.add(layers.Conv2D(64, kernel_size=5, strides=2, input_shape=img_shape, padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(dropout))
    model.add(layers.Conv2D(128, kernel_size=5, strides=2, padding="same"))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(dropout))
    model.add(layers.Conv2D(256, kernel_size=5, strides=2, padding="same"))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(dropout))
    model.add(layers.Conv2D(512, kernel_size=5, strides=1, padding="same"))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(dropout))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    model.summary()
    img = models.Input(shape=img_shape)
    validity = model(img)
    return models.Model(img, validity)


# 从文件夹加载图片数据
def load_dir_img(sorcedir):
    print('正在读取图片...')
    files = os.listdir(sorcedir)
    data = np.zeros((files.__len__(),) + image.img_to_array(image.load_img(os.path.join(sorcedir, files[0]))).shape)
    for i in range(files.__len__()):
        data[i] = image.img_to_array(image.load_img(os.path.join(sorcedir, files[i]))) / 127.5 - 1
    return data


# ************************** 训练
"""
    gdrate:额外的生成器训练比率(判别器50%额外训练0次,100%额外训练gdrate次)
    save_interval:保存间隔(steap)
"""
def run(epochs=100, batch_size=128, save_interval=100, gdrate=3, save_dir='.\\gan_image', history=None):
    last_time = time.clock()
    start_epoch = 0
    if history is None:
        history = []
    else:
        start_epoch = int(history[-1][0])
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))
    for epoch in range(epochs):
        for step in range(x.shape[0] // batch_size):
            # 按q终止
            while msvcrt.kbhit():
                char = ord(msvcrt.getch())
                if char == 113:
                    return history
            g_loss = -1
            # 训练判别器
            imgs = x[step * batch_size:step * batch_size + batch_size]
            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            gen_imgs = generator.predict(noise)
            d_loss_real = discriminator.train_on_batch(imgs, valid)
            d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            # 训练生成器(动态训练比例)
            for i in range(1 + int(gdrate * np.maximum(d_loss[1] - .5, 0) * 2)):
                noise = np.random.normal(0, 1, (batch_size, latent_dim))
                g_loss = combined.train_on_batch(noise, valid)
            # Log
            if step % save_interval == 0:
                print(
                    "%d:%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch+start_epoch, step, d_loss[0], 100 * d_loss[1], g_loss))
                history.append([epoch+start_epoch, step, d_loss[0], 100 * d_loss[1], g_loss])
                combined.save('gan.h5')
                # 保存生成的图像
                img = image.array_to_img(gen_imgs[0] * 127 + 127., scale=False)
                img.save(os.path.join(save_dir, 'train_' + str(epoch+start_epoch) + '_' + str(step) + '.png'))
                # 保存真实图像，以便进行比较
                # img = image.array_to_img(imgs[0] * 127 + 127., scale=False)
                # img.save(os.path.join(save_dir, 'real_' + str(epoch+start_epoch) + '_' + str(step) + '.png'))
        # 计时
        print('epoch run %d s, total run %d s' % (time.clock() - last_time, time.clock()))
        last_time = time.clock()
    combined.save('gan.h5')
    return history


# ************************** 生成
def generate(generator, num=100, save_dir=r'gan_image'):
    noise = np.random.normal(0, 1, (num, K.int_shape(generator.layers[0].input)[1]))
    gen_imgs = generator.predict(noise)
    for i in range(gen_imgs.shape[0]):
        img = image.array_to_img(gen_imgs[i] * 127 + 127., scale=False)
        img.save(os.path.join(save_dir, 'generated_' + str(i) + '.png'))

# ************************** 中途保存
def save(folder):
    combined.save(os.path.join(folder, 'gan.h5'))
    generator.save(os.path.join(folder, 'gan_g.h5'))
    discriminator.save(os.path.join(folder, 'gan_d.h5'))
    np.save(os.path.join(folder, 'history.npy'), history)

def load(folder):
    history = np.load(os.path.join(folder, 'history.npy')).tolist()
    generator = models.load_model(os.path.join(folder, 'gan_g.h5'))
    discriminator = models.load_model(os.path.join(folder, 'gan_d.h5'))
    discriminator.trainable = False
    input_noise = models.Input(shape=(latent_dim,))
    optimizer = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    combined = models.Model(input_noise, discriminator(generator(input_noise)))
    combined.compile(loss='binary_crossentropy', optimizer=optimizer)
    return history, generator, discriminator, combined


# ************************** Load Data
# 数据来源:https://drive.google.com/drive/folders/1mCsY5LEsgCnc0Txv0rpAUhKVPWVkbw5I?usp=sharing
# x = load_dir_img(r'C:\dataset\faces3m96')
print('正在加载数据')
x = np.load(r'C:\Users\78753\Desktop\DL\picpro\faces5m96.npy')


# ************************** 建模
optimizer = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
# 对判别器进行构建和编译
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
# 对生成器进行构造
generator = build_generator()
# 构造对抗模型
# 总体模型只对生成器进行训练
discriminator.trainable = False
input_noise = models.Input(shape=(latent_dim,))
combined = models.Model(input_noise, discriminator(generator(input_noise)))
combined.compile(loss='binary_crossentropy', optimizer=optimizer)


# ************************** 运行
history = run()

# history, generator, discriminator, combined=load(r'C:\Users\78753\Desktop\DL\picpro\moegirlgandone52g+')
# history = run(history=history)
