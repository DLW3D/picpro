# coding=gbk
import os
import csv
import random

import keras
from PIL import Image
import numpy as np

from AutoCrop import AutoCrop


def ReadImageToArray(filename, usefilter=True):
    # ��ȡͼƬ
    try:
        im = Image.open(filename)
    except(OSError, NameError):
        print("OSError On:" + filename)
        return None
    # ��ʾͼƬ
    # im.show()
    im = im.convert("RGBA")
    data = np.array(im)
    # ����͸������RGBAתRGB
    clear = data[:, :, 3] != 0
    data = data[:, :, 0:3]
    if usefilter:
        for i in range(data.shape[2]):
            data[:, :, i] *= clear
    return data
    # ����
    # new_im = Image.fromarray(np.uint8(data))
    # new_im.show()


def ArrayToImage(data):
    if data is None:
        return None
    new_im = Image.fromarray(data.astype(np.uint8))
    return new_im


# �ü�������ɫ����
def Crop(img):
    return AutoCrop(img, backgroundColor=(0, 0, 0))


# ͼƬ����
def Scale(img, width, height, usecrop=True, blackedge=0):
    if img is None:
        return None
    # �ü�������ɫ����
    if usecrop:
        img = Crop(img)
    # ������ڱ�%
    if blackedge != 0:
        w, h = img.size
        img = img.crop(
            (-w * blackedge * random.random(), -h * blackedge * random.random(), w + w * blackedge * random.random(),
             h + h * blackedge * random.random()))
    # �ȱ�������
    w, h = img.size
    if w / h > width / height:
        h = int(h * width / w)
        w = width
    else:
        w = int(w * height / h)
        h = height
    img = img.resize((w, h), Image.ANTIALIAS)
    # �ü�
    x0 = int((w - width) / 2)
    y0 = int((h - height) / 2)
    x1 = x0 + width
    y1 = y0 + height
    img = img.crop((x0, y0, x1, y1))
    return img


# ͼƬ���� file2file
def Process(filename, savename, w, h):
    img = Scale(ArrayToImage(ReadImageToArray(filename)), w, h)
    if img is not None:
        img.save(savename)


# ͼƬ���� ������folder2folder************
def Processes(sorcedir, targetdir, w, h):
    print("����ɨ���ļ���...")
    # �г�Ŀ¼���ļ�����Ŀ¼
    files = os.listdir(sorcedir)
    # ɸѡ���ļ�
    files = [f for f in files if os.path.isfile(os.path.join(sorcedir, f))]
    print("���ڴ����ļ�...")
    for f in files:
        Process(os.path.join(sorcedir, f), os.path.join(targetdir, f), w, h)
    return


# ��ȡ�ļ���������ͼƬ,���Array
def ReadDatas(sorcedir):
    print('���ڶ�ȡͼƬ...')
    files = os.listdir(sorcedir)
    files = [f for f in files if os.path.isfile(os.path.join(sorcedir, f))]
    data = []
    for f in files:
        arr = ReadImageToArray(os.path.join(sorcedir, f))
        if arr is not None:
            data.append(arr)
    return np.array(data, dtype=int)


# ��Array�����CSV (�ļ��Ĵ�С�����ͼƬ��20������)
def SaveFile(label, data, path):
    print('���ڱ����CSV�ļ�...')
    # reshape
    shape = data.shape
    m = shape[0]  # ����
    pix = np.prod(shape[1:])  # ����
    data.shape = (m, pix)
    # ʹ�ÿո�ָ�
    data = np.array(data, dtype=str)
    newdata = []
    for i in range(m):
        newdata.append(' '.join(data[i]))
    # д�ļ�
    with open(path, 'w', newline='')as f:
        f_csv = csv.writer(f)
        f_csv.writerow(['Label', 'Feature'])  # ��ͷ
        f_csv.writerows(np.concatenate([np.ones((m, 1), dtype=int) * label,
                                        np.array(newdata, dtype=str).reshape(m, 1)], 1))  # ����
    print('д�����!')
    return


# ���ļ���������ͼƬ���� ͼƬ���� �����CSV�ļ�*********
def FileToCSV(label, dir, path, width=0, height=0):
    if width == 0 | height == 0:
        SaveFile(label, ReadDatas(dir), path)
    else:
        print("����ɨ���ļ���...")
        # �г�Ŀ¼���ļ�����Ŀ¼
        files = os.listdir(dir)
        # ɸѡ���ļ�
        files = [f for f in files if os.path.isfile(os.path.join(dir, f))]
        print("���ڴ����ļ�...")
        data = []
        for f in files:
            img = Scale(ArrayToImage(ReadImageToArray(os.path.join(dir, f))), width, height)
            if img is not None:
                data.append(np.array(img, dtype=int))
        SaveFile(label, np.array(data, dtype=int), path)
    return


# ��ȡCSV
def ReadFile(path, rate=10, shape=(100,100,3)):
    print("���ڶ�ȡCSV�ļ�...")
    x_train = []
    x_label = []
    val_data = []
    val_label = []
    raw_train = np.genfromtxt(path, delimiter=',', dtype=str, skip_header=1)
    for i in range(len(raw_train)):
        tmp = np.array(raw_train[i, 1].split(' ')).reshape(shape)
        if (i % rate == 0):
            val_data.append(tmp)
            val_label.append(raw_train[i][0])
        else:
            x_train.append(tmp)
            x_train.append(np.flip(tmp, axis=2))  # simple example of data augmentation
            x_label.append(raw_train[i][0])
            x_label.append(raw_train[i][0])
    x_train = np.array(x_train, dtype=float) / 255.0
    val_data = np.array(val_data, dtype=float) / 255.0
    x_label = np.array(x_label, dtype=int)
    val_label = np.array(val_label, dtype=int)
    return x_train, x_label, val_data, val_label


# ʹ��ģ���ж�ͼƬ*********
def Tester(model, path):
    tester = np.array(Scale(ArrayToImage(ReadImageToArray(path)), 100, 100)).reshape(1, 100, 100, 3) / 255
    print(model.predict(tester))


# Tester(model, 1, r'C:\Users\78753\Desktop\1.png')
# ArrayToImage(x_train[0]*255+np.random.rand(100,100,3)*10).show()

# a,b,c,d = ReadFile('C:\\Users\\78753\\Desktop\\��R\\1.csv')

# data = ReadDatas("C:\\Users\\78753\\Desktop\\��R\\tar")
# SaveFile(1,data,"C:\\Users\\78753\\Desktop\\��R\\1.csv")

# a = ReadDatas('C:\\Users\\78753\\Desktop\\��R\\����')
# print(a)

# filename = '..\\��R\\sum\\199 ������ˮ����԰.png'
# data = ReadImageToArray(filename)
# # print(data)
# new_im = ArrayToImage(data)
# # plt.imshow(data, cmap=plt.cm.gray, interpolation='nearest')
# new_im.resize((500, 500), Image.ANTIALIAS)
# new_im.show()
# new_im.save('lena_1.png')

# ArrayToImage(ReadImageToArray('..\\��R\\sum\\011 ʥ����.PNG')).show()
# Scale(Image.open('akasita_2-pic.png'),100,100).paste((20,20,80,80)).show()