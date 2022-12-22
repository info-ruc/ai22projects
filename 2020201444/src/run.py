#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Version : Python 3.9
import pickle

import torch
from torch import nn,optim
from config_utils import Config,NewDataloader
from model import MyModel
from tqdm import tqdm
from PIL import Image
import numpy as np

class Runner(object):
    def __init__(self,loader,config):
        self.loader=loader
        self.config=config
        self.model=MyModel()
        self.loss = nn.MSELoss()

    def train(self):
        train_loader, test_data = self.loader
        optimizer = optim.Adam(self.model.parameters(), lr=config.lr)
        print(self.model)
        print('traning model parameters:')
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print('%s :  %s' % (name, str(param.data.shape)))
        print('--------------------------------------')
        print('start to train the model ...')
        e_min = 1.0
        for epoch in range(1, 1 + self.config.epoch):
            train_loss = 0.0
            data_iterator = tqdm(train_loader, desc='Train')
            for step, train_data in enumerate(data_iterator):
                self.model.train()
                train_data = train_data.to(self.config.device)
                optimizer.zero_grad()
                x1_code,x2_code,delta_x1,x1_old,x1_new, x2_old, x2_new,x1_next=self. model(train_data)
                print("重构损失:",self.loss(x1_old, x1_new)+self.loss(x1_next,x2_old))
                print("时序依赖损失:",self.loss(x1_code + delta_x1, x2_code))
                loss=self.loss(x1_old, x1_new)+self.loss(x1_next,x2_old)+self.loss(x1_code + delta_x1, x2_code)
                print(loss)
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
            print('Epoch {} Loss:{:.3f}'.format(epoch + 1, train_loss / len(train_loader)))

            first_test_data = torch.clone(test_data[0:1])
            temp_test_data = first_test_data
            test_pre_data = torch.zeros((100, 3, 128, 128), dtype=torch.float32)
            test_pre_data[0] = temp_test_data[0][0]

            with torch.no_grad():
                self.model.eval()
                for i in range(99):
                    x1_code, x2_code, delta_x1, x1_old, x1_new, x2_old, x2_new, x1_next = self.model(temp_test_data)
                    test_pre_data[i + 1] = x1_next[0]
                    temp_test_data[0][0] = x1_next
            test_real_data = torch.cat([test_data[0:1, 0, :, :, :], test_data[:, 1, :, :, :]], dim=0)
            e = torch.sqrt(torch.square(
                        torch.norm(torch.flatten(test_pre_data[0:100] - test_real_data[0:100], start_dim=0),
                                   p=2) / torch.norm(torch.flatten(test_real_data[0:100], start_dim=0), p=2)))
            print("e: {}".format(e))
            if(e_min>e):
                e_min=e
                torch.save(self.model.state_dict(), self.config.model_dir)
                print('>>> save models!')
            e_list = []
            # for i in range(100):
            #     e = torch.sqrt(torch.square(
            #         torch.norm(torch.flatten(test_pre_data[0:i + 1] - test_real_data[0:i + 1], start_dim=0),
            #                    p=2) / torch.norm(torch.flatten(test_real_data[0:i + 1], start_dim=0), p=2)))
            #     e_list.append(e.item())

    def test(self):
        print('-------------------------------------')
        print('start test ...')
        print(self.model)
        _, test_data = self.loader
        self.model.load_state_dict(torch.load(self.config.model_dir))
        first_test_data=torch.clone(test_data[0:1])
        temp_test_data=first_test_data
        test_pre_data=torch.zeros((100,3,128,128),dtype=torch.float32)
        test_pre_data[0]=temp_test_data[0][0]

        with torch.no_grad():
            self.model.eval()
            for i in range(99):
                x1_code,x2_code,delta_x1,x1_old,x1_new, x2_old, x2_new,x1_next=self.model(temp_test_data)
                test_pre_data[i+1]=x1_next[0]
                temp_test_data[0][0]=x1_next
        test_real_data=torch.cat([test_data[0:1,0,:,:,:],test_data[:,1,:,:,:]],dim=0)
        e_list=[]
        for i in range(100):
            e=torch.sqrt(torch.square(torch.norm(torch.flatten(test_pre_data[0:i+1]-test_real_data[0:i+1],start_dim=0),p=2)/torch.norm(torch.flatten(test_real_data[0:i+1],start_dim=0),p=2)))
            e_list.append(e.item())
        print(e_list)
        test_pre_data=(test_pre_data*255).numpy()

        fin=open(self.config.result_dir,"wb")
        pickle.dump(test_pre_data,fin)
        fin.close()



if __name__ == '__main__':
    config=Config()
    print('--------------------------------------')
    print('some config:')
    config.print_config()

    print('--------------------------------------')
    print('start to load data ...')
    loader=NewDataloader(config)
    train_loader, test_data = None, None
    if config.mode == 0:
        train_loader = loader.get_train()
        test_data = loader.get_test()
    elif config.mode == 1:
        test_data = loader.get_test()

    loader = [train_loader, test_data]
    print('finish!')

    runner = Runner(loader, config)
    if config.mode == 0:
        runner.train()
        runner.test()
    elif config.mode == 1:
        runner.test()
    else:
        raise ValueError('invalid train mode!')
