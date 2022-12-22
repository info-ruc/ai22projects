import os
from PIL import Image
import numpy as np

base_path = 'E:/ai/'
train_path = 'E:/ai/train_data/'
test_path = 'E:/ai/test_data/'
origin_path = 'E:/ai/origin/'

#读取图片进入filelist
def read_image(paths):
    os.listdir(paths)
    filelist = []
    for root, dirs, files in os.walk(paths):
        for file in files:
            if os.path.splitext(file)[1] == ".jpg":
                filelist.append(root + '/' + file)
    return filelist
#另存为图片
def im_xiangsu(paths,classname):
    for filename in paths:
        try:
            path = train_path + classname
            path = path.strip()
            isExists = os.path.exists(path)
            if not isExists:
                os.makedirs(path)
                # print('创建新文件夹' + classname)
            # print('point1')
            #print(filename)
            im = Image.open(filename)
            newim = im.resize((128, 128))
            # print('point2')
            #print(path + '/' + filename[(len(origin_path) + 1):-4] + '.jpg')
            newim.save(path + '/' + filename[(len(origin_path) + 1 + len(classname)):-4] + '.jpg')
            # print('图片' + filename[12:-4] + '.jpg' + '像素转化完成')
        except OSError as e:
            print(e.args)

#图片数据转化为数组
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

for root, dirs, files in os.walk(origin_path):
    for dir in dirs:
        lists = dir.split('/')
        path_list.append(lists[-1])
        print(lists[-1])

for classname in path_list:
    file_list = read_image(origin_path + classname)
    filelist_list.append(file_list)

for i in range(len(filelist_list)):
    im_xiangsu(filelist_list[i],path_list[i])

# path_1 = origin_path  + 'dandelion'
# path_2 = origin_path  + 'roses'
# print(path_1)
# print(path_2)
# filelist_1 = read_image(path_1)
# filelist_2 = read_image(path_2)
# print(filelist_1)
# print(filelist_2)
# im_xiangsu(filelist_1,'dandelion')
# im_xiangsu(filelist_2,'roses')
