#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Version : Python 3.9

from PIL import Image
import numpy as np
import os
import pickle

def load_image(path):
    image=Image.open(path)
    image_np=np.array(image)/255
    return np.expand_dims(np.transpose(image_np,(2,0,1)),axis=0)

def preprocess_data(src_path,des_path,file_type):
    des_path_new=os.path.join(des_path,"{}.pkl".format(file_type))
    image_np_2_list=[]
    if(file_type=="train"):
        for i in range(10):
            path_instance=os.path.join(src_path,str(i))
            image_np_list=[]
            for j in range(100):
                image_np=load_image(os.path.join(path_instance,"{}.png".format(str(j))))
                image_np_list.append(image_np)
            for j in range(99):
                image_np_2_list.append(np.concatenate((image_np_list[j],image_np_list[j+1]),axis=0))
    if(file_type=="test"):
        path_instance = os.path.join(src_path, str(10))
        image_np_list = []
        for j in range(100):
            image_np = load_image(os.path.join(path_instance, "{}.png".format(str(j))))
            image_np_list.append(image_np)
        for j in range(99):
            image_np_2_list.append(np.concatenate((image_np_list[j], image_np_list[j + 1]), axis=0))
    image_np_2=np.stack(image_np_2_list)
    image_list_2=image_np_2.tolist()
    fout=open(des_path_new,"wb")
    pickle.dump(image_list_2,fout)
    fout.close()
    return image_list_2





if __name__ == '__main__':
    src_dir= 'data/diffusion_of_reaction'
    des_dir='./processed'
    if not os.path.exists(des_dir):
        os.makedirs(des_dir)
    preprocess_data(src_dir,des_dir,"train")
    preprocess_data(src_dir,des_dir,"test")

