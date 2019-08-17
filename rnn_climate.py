# ***************** 读取数据
import os
import numpy as np

# https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv.zip
data_dir = r'D:\78753\.keras\data'
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')

f = open(fname)
data = f.read()
f.close()

lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]

print(header)
print(len(lines))

float_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    float_data[i, :] = [float(x) for x in line.split(',')[1:]]

# ************* 检查数据
from matplotlib import pyplot as plt

# 显示2009-2016温度变化图
temp = float_data[:, 1]
plt.plot(range(len(temp)), temp)
plt.show()

# 前10天温度变化图(十分钟一个记录,144记录/天)
plt.plot(range(1440), temp[:1440])
plt.show()

# ************* 数据生成器
'''
data:浮点数据的原始数组，我们刚刚在上面的代码片段中将其规范化。
lookback:我们的输入数据应该返回多少个时间步骤。
delay:未来我们的目标应该是多少个时间步骤。
min_index和max_index:指数data数组，该数组分隔要从哪个时间步骤绘制的时间步骤。这对于保存一段数据以进行验证和另一段数据用于测试非常有用。
shuffle:是洗牌我们的样品，还是按时间顺序提取样品。
batch_size:每批样品的数量。
step:按时间步骤对数据进行抽样的时间。我们将设置为6，以便每小时绘制一个数据点。

假设现在是1点，我们要预测2点时的气温，由于当前数据记录的是每隔10分钟时的气象数据，1点到2点
间隔1小时，对应6个10分钟，这个6对应的就是delay

要训练网络预测温度，就需要将气象数据与温度建立起对应关系，我们可以从1点开始倒推10天，从过去
10天的气象数据中做抽样后，形成训练数据。由于气象数据是每10分钟记录一次，因此倒推10天就是从
当前时刻开始回溯1440条数据，这个1440对应的就是lookback

我们无需把全部1440条数据作为训练数据，而是从这些数据中抽样，每隔6条取一条，
因此有1440/6=240条数据会作为训练数据，这就是代码中的lookback//step

于是我就把1点前10天内的抽样数据作为训练数据，2点是的气温作为数据对应的正确答案，由此
可以对网络进行训练
'''
def generator(data, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)
        samples = np.zeros((len(rows),
                           lookback // step,
                           data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets


# ************** 调用数据生成器
# 选用前200,000个时间戳作为培训数据
# 减去均值除以标准差
mean = float_data[:200000].mean(axis=0)
float_data -= mean
std = float_data[:200000].std(axis=0)
float_data /= std

lookback = 1440   # 观察将追溯到5天前。
step = 6   # 观测将在每小时一个数据点取样。
delay = 144   # 目标是未来24小时。
batch_size = 128

train_gen = generator(float_data,
                      lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=200000,
                      shuffle=True,
                      step=step,
                      batch_size=batch_size)
val_gen = generator(float_data,
                    lookback=lookback,
                    delay=delay,
                    min_index=200001,
                    max_index=300000,
                    step=step,
                    batch_size=batch_size)
test_gen = generator(float_data,
                     lookback=lookback,
                     delay=delay,
                     min_index=300001,
                     max_index=None,
                     step=step,
                     batch_size=batch_size)

# 要遍历整个验证集需要的gen调用次数
val_steps = (300000 - 200001 - lookback) // batch_size
test_steps = (len(float_data) - 300001 - lookback) // batch_size

# 常识基线评估(总是预测明天某时的温度等于今天某时的温度的损失)
def evaluate_naive_method():
    batch_maes = []
    for step in range(val_steps):
        samples, targets = next(val_gen)
        preds = samples[:, -1, 1]
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)
    print(np.mean(batch_maes))

evaluate_naive_method()
# 0.2897359729905486

# **************** 建模
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
# **************** 建模(Dense)0.30
# model = Sequential()
# model.add(layers.Flatten(input_shape=(lookback // step, float_data.shape[-1])))
# model.add(layers.Dense(32, activation='relu'))
# model.add(layers.Dense(1))
#
# model.compile(optimizer=RMSprop(), loss='mae')
# history = model.fit_generator(train_gen,
#                               steps_per_epoch=500,
#                               epochs=20,
#                               validation_data=val_gen,
#                               validation_steps=val_steps)

# **************** 建模(GRU)0.2640
# model = Sequential()
# model.add(layers.GRU(32, input_shape=(None, float_data.shape[-1])))
# model.add(layers.Dense(1))
#
# model.compile(optimizer=RMSprop(), loss='mae')
# history = model.fit_generator(train_gen,
#                               steps_per_epoch=500,
#                               epochs=20,
#                               validation_data=val_gen,
#                               validation_steps=val_steps)

# **************** 建模(GRU,DroupOut)0.2619
# model = Sequential()
# model.add(layers.GRU(32,
#                      dropout=0.2,
#                      recurrent_dropout=0.2,
#                      input_shape=(None, float_data.shape[-1])))
# model.add(layers.Dense(1))
#
# model.compile(optimizer=RMSprop(), loss='mae')
# history = model.fit_generator(train_gen,
#                               steps_per_epoch=500,
#                               epochs=40,
#                               validation_data=val_gen,
#                               validation_steps=val_steps)

# **************** 建模(GRU,DroupOut,Deep)0.2633
model = Sequential()
model.add(layers.GRU(32,
                     dropout=0.1,
                     recurrent_dropout=0.5,
                     return_sequences=True,
                     input_shape=(None, float_data.shape[-1])))
model.add(layers.GRU(64, activation='relu',
                     dropout=0.1,
                     recurrent_dropout=0.5))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs=40,
                              validation_data=val_gen,
                              validation_steps=val_steps)




# ***************** 显示训练曲线
import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

