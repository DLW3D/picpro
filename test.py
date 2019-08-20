import os
import tensorflow as tf
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

latent_dim = 100
img_shape = (96, 96, 3)


def build_generator():
    model = Sequential()
    model.add(layers.Dense(512 * 6 * 6, activation='relu', input_dim=latent_dim))  # 输入维度为100
    model.add(layers.Reshape((6, 6, 512)))
    model.add(layers.UpSampling2D())  # 进行上采样，变成14*14*128
    model.add(layers.Conv2D(256, kernel_size=5, padding='same'))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Activation("relu"))  #
    model.add(layers.UpSampling2D())
    model.add(layers.Conv2D(128, kernel_size=5, padding="same"))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Activation("relu"))
    model.add(layers.UpSampling2D())
    model.add(layers.Conv2D(64, kernel_size=5, padding="same"))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Activation("relu"))
    model.add(layers.UpSampling2D())
    model.add(layers.Conv2D(img_shape[-1], kernel_size=5, padding="same"))
    model.add(layers.Activation("tanh"))
    model.summary()  # 打印网络参数
    noise = models.Input(shape=(latent_dim,))
    img = model(noise)
    return models.Model(noise, img)  # 定义一个 一个输入noise一个输出img的模型


def build_discriminator():
    model = Sequential()
    dropout = 0.4
    model.add(layers.Conv2D(64, kernel_size=5, strides=2, input_shape=img_shape, padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(dropout))
    model.add(layers.Conv2D(128, kernel_size=5, strides=2, padding="same"))
    model.add(layers.ZeroPadding2D(padding=((0, 1), (0, 1))))
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


def load_dir_img(sorcedir):
    print('正在读取图片...')
    files = os.listdir(sorcedir)
    data = []
    for f in files:
        arr = image.img_to_array(image.load_img(os.path.join(sorcedir, f)))
        data.append(arr)
    return np.array(data) / 127.5 - 1


# ***************************** 建模
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

# ************************* Load Data
epochs = 100
batch_size = 64
gdrate = 2  # 训练生成器额外训练倍率
save_interval = 10
save_dir = '.\\gan_image'

data_dir = r'C:\Users\78753\.keras\data\2faces\96\faces'
x = load_dir_img(data_dir)
# train_datagen = ImageDataGenerator(rescale=1./255)
# train_generator = train_datagen.flow_from_directory(
#     # 目标文件夹
#     data_dir,
#     # 规范化图片大小
#     target_size=img_shape[0:2],
#     batch_size=batch_size)

# *************************** 训练
valid = np.ones((batch_size, 1))
fake = np.zeros((batch_size, 1))
for epoch in range(epochs):
    for step in range(x.shape[0] // batch_size):
        g_loss = -1
        # 训练判别器
        imgs = x[step * batch_size:step * batch_size + batch_size]
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        gen_imgs = generator.predict(noise)
        d_loss_real = discriminator.train_on_batch(imgs, valid)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        # 训练生成器(动态训练比例)
        for i in range(1+int(gdrate * d_loss[1] * 2)):
            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            g_loss = combined.train_on_batch(noise, valid)
        # Log
        if step % save_interval == 0:
            print("%d:%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, step, d_loss[0], 100 * d_loss[1], g_loss))
            combined.save('gan.h5')
            # 保存生成的图像
            img = image.array_to_img(gen_imgs[0] * 127 + 127., scale=False)
            img.save(os.path.join(save_dir, 'generated_' + str(epoch) + '_' + str(step) + '.png'))
            # 保存真实图像，以便进行比较
            # img = image.array_to_img(imgs[0] * 127 + 127., scale=False)
            # img.save(os.path.join(save_dir, 'real_' + str(epoch) + '_' + str(step) + '.png'))

combined.save('gan.h5')

