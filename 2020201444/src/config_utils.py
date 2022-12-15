#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Version : Python 3.9

import pickle
import torch
import os
from torch.utils.data import Dataset, DataLoader

class Config(object):
    def __init__(self):
        self.epoch=200
        self.batch_size=32
        self.lr=0.001
        self.mode=0
        self.data_dir="./processed"
        self.output_dir="./output"

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.model_dir = os.path.join(self.output_dir, "model.pkl")
        self.result_dir=os.path.join(self.output_dir,"result.pkl")
        self.image_dir=os.path.join(self.output_dir,"image")
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)
        self.cuda=-1
        self.device = None
        if self.cuda >= 0 and torch.cuda.is_available():
            self.device = torch.device('cuda:{}'.format(self.cuda))
        else:
            self.device = torch.device('cpu')

    def print_config(self):
        for key in self.__dict__:
            print(key, end=' = ')
            print(self.__dict__[key])

class NewDataset(Dataset):
    def __init__(self, data):
        self.dataset = data

    def __getitem__(self, index):
        data = self.dataset[index]
        return data

    def __len__(self):
        return len(self.dataset)

class NewDataloader(object):
    def __init__(self,config):
        self.data_dir=config.data_dir
        self.batch_size=config.batch_size

    def _get_data(self,file_type,shuffle=False):
        fin=open(os.path.join(self.data_dir,"{}.pkl".format(file_type)),"rb")
        data=pickle.load(fin)
        data=torch.tensor(data).to(torch.float32)
        data_set=NewDataset(data)
        loader=DataLoader(data_set,batch_size=self.batch_size,shuffle=shuffle)
        return loader

    def get_train(self):
        train_loader = self._get_data(file_type='train', shuffle=True)
        print('finish loading train!')
        return train_loader

    def get_test(self):
        # test_loader = self._get_data(file_type='test', shuffle=False)
        # print('finish loading test!')
        fin = open(os.path.join(self.data_dir, "test.pkl"), "rb")
        data=pickle.load(fin)
        test_data = torch.tensor(data).to(torch.float32)
        return test_data


if __name__ == '__main__':
    config=Config()
    config.print_config()
    loader=NewDataloader(config)
    train_loader = loader.get_train()
    for step, train_data in enumerate(train_loader):
        print(step,train_data.shape)
        break
