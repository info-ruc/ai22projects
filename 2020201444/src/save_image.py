#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Version : Python 3.9
import cv2
import numpy as np
import pickle
from config_utils import Config
from PIL import Image
import os


def save_image(config):
    fout=open(config.result_dir,"rb")
    test_pre_data=pickle.load(fout)
    fout.close()
    for i in range(test_pre_data.shape[0]):
        image_path_i=os.path.join(config.image_dir,"{}.png".format(str(i)))
        x=np.array(np.transpose(test_pre_data[i],(1,2,0)),dtype=np.uint8)
        img=Image.fromarray(x,'RGB')
        img.save(image_path_i)


if __name__ == '__main__':
    config=Config()
    save_image(config)

