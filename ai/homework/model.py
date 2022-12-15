import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

base_path = 'E:/ai/'
train_path = 'E:/ai/train_data/'
test_path = 'E:/ai/test_data/'
origin_path = 'E:/ai/origin/'

def read_image(paths):
    os.listdir(paths)
    filelist = []
    for root, dirs, files in os.walk(paths):
        for file in files:
            if os.path.splitext(file)[1] == ".jpg":
                filelist.append(os.path.join(root, file))
    return filelist

def im_array(paths):
    M=[]
    for filename in paths:
        im=Image.open(filename)
        im_L=im.convert("L")                #模式L
        Core=im_L.getdata()
        arr1=np.array(Core,dtype='float32')/255.0
        list_img=arr1.tolist()
        M.extend(list_img)
    return M

path_list = []
filelist_list = []

for root, dirs, files in os.walk(train_path):
    for dir in dirs:
        lists = dir.split('/')
        path_list.append(lists[-1])
        print(lists[-1])

for classname in path_list:
    file_list = read_image(train_path + classname)
    filelist_list.append(file_list)

dict_label = {}
num = 0
for root, dirs, files in os.walk(train_path):
    for dir in dirs:
        listword = dir.split('/')
        dict_label[num] = listword[-1]
        num += 1

# path_1 = train_path + 'dandelion'
# path_2 = train_path + 'roses'
# filelist_1 = read_image(path_1)
# filelist_2 = read_image(path_2)
filelist_all = []
for i in range(len(filelist_list)):
    filelist_all.extend(filelist_list[i])
# filelist_all = filelist_1 + filelist_2
M = im_array(filelist_all)
# dict_label={0:'dandelion',1:'roses'}
train_images=np.array(M).reshape(len(filelist_all),128,128)
# label=[0]*len(filelist_1)+[1]*len(filelist_2)
label = []
for i in range(len(filelist_list)):
    label.extend([i]*len(filelist_list[i]))
train_lables=np.array(label)        #数据标签
train_images = train_images[ ..., np.newaxis ]        #数据图片
print(train_images.shape)
number = len(filelist_list)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)))#过滤器个数，卷积核尺寸，激活函数，输入形状
model.add(layers.MaxPooling2D((2, 2)))#池化层
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())#降维
model.add(layers.Dense(64, activation='relu'))#全连接层
model.add(layers.Dense(number, activation='softmax'))
model.summary()  # 显示模型的架构
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#epochs为训练多少轮、batch_size为每次训练多少个样本
model.fit(train_images, train_lables, epochs=5)
model.save('my_model.h5') #保存为h5模型
print("模型保存成功！")