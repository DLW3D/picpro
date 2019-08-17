
# ************* 加载数据
import os
# download from http://ai.stanford.edu/~amaas/data/sentiment/
imdb_dir = r'C:\Users\78753\.keras\data\aclImdb'
train_dir = os.path.join(imdb_dir, 'train')

labels = []
texts = []

for label_type in ['neg', 'pos']:
    dir_name = os.path.join(train_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname),'r', encoding='UTF-8')
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)

# ************* 格式化数据
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

maxlen = 100  # 一条评论最大长度 100 words
training_samples = 20000  # 训练样本 200 samples
validation_samples = 5000  # 验证样本 10000 samples
max_words = 10000  # 只编码最多 10000 words

# 初始化分词器
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
# 使用分词器编码文本
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

# 截断过长样本
data = pad_sequences(sequences, maxlen=maxlen)

labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# 打乱数据
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

# 拆分数据
x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]


# ************* 调用GloVe
# download from https://nlp.stanford.edu/projects/glove/
glove_dir = r'C:\Users\78753\.keras\data\glove'

embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'),'r', encoding='UTF-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs

f.close()

print('Found %s word vectors.' % len(embeddings_index))

# GloVe的向量维度
embedding_dim = 100

# 匹配GloVe向量
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector


# ************** 建模
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

model = Sequential()
# input (?, maxlen)
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
# (?, maxlen, embedding_dim)
model.add(Flatten())
# (?, maxlen * embedding_dim)
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# 手动装配权重
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

# 训练
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_val, y_val))
model.save('save.h5')


# ************* 测试
test_dir = os.path.join(imdb_dir, 'test')

labels = []
texts = []

for label_type in ['neg', 'pos']:
    dir_name = os.path.join(test_dir, label_type)
    for fname in sorted(os.listdir(dir_name)):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname),'r', encoding='UTF-8')
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)

sequences = tokenizer.texts_to_sequences(texts)
x_test = pad_sequences(sequences, maxlen=maxlen)
y_test = np.asarray(labels)

model.evaluate(x_test, y_test)


