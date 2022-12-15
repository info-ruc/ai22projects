import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

model= tf.keras.models.load_model('my_model.h5')
model.summary() #网络结构
print("模型加载完成！")

#读取图片进入filelist
def read_image(paths):
    os.listdir(paths)
    filelist = []
    for root, dirs, files in os.walk(paths):
        for file in files:
            if os.path.splitext(file)[1] == ".jpg":
                filelist.append(os.path.join(root, file))
    return filelist
#另存为图片
def im_xiangsu(paths,classname):
    for filename in paths:
        try:
            path = test_path + classname
            path = path.strip()
            isExists = os.path.exists(path)
            if not isExists:
                os.makedirs(path)
                print('创建新文件夹' + classname)
            # print('point1')
            # print(filename)
            im = Image.open(filename)
            newim = im.resize((128, 128))
            # print('point2')
            # print(path + '/' + filename[(len(test_path) + 1):-4] + '.jpg')
            newim.save(path + '/' + filename[(len(test_path) + 1 + len('origin')):-4] + '.jpg')
            # print('图片' + filename[12:-4] + '.jpg' + '像素转化完成')
        except OSError as e:
            print(e.args)

def im_array(paths):
    im = Image.open(paths[0])
    im_L = im.convert("L")  # 模式L
    Core = im_L.getdata()
    arr1 = np.array(Core, dtype='float32') / 255.0
    list_img = arr1.tolist()
    images = np.array(list_img).reshape(-1,128, 128,1)
    return images

test_path = 'E:/ai/test_data/'
origin_path = 'E:/ai/origin/'
dict_label = {}
path_list = []
num = 0
for root, dirs, files in os.walk(origin_path):
    for dir in dirs:
        listword = dir.split('/')
        dict_label[num] = listword[-1]
        num += 1

# print(dict_label)
# dict_label={0:'dandelion',1:'roses'}
o_test = test_path + 'origin/'
filelist=read_image(o_test)
im_xiangsu(filelist,'new')
filelist=read_image(test_path + 'new/')
img=im_array(filelist)
#预测图像
predictions_single=model.predict(img)
print("预测结果为:",dict_label[np.argmax(predictions_single)])
#返回数组中概率最大的那个
# print(predictions_single)
