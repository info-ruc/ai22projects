#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Version : Python 3.9

import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3,8,3,stride=1,padding=1), # 8*128*128
            nn.ReLU(True),
            nn.MaxPool2d(2, 2), #8*64*64
            nn.BatchNorm2d(8),
            nn.Conv2d(8,16,3,stride=1,padding=1),  #16*64*64
            nn.ReLU(True),
            nn.BatchNorm2d(16),
            nn.Conv2d(16,32,3,stride=1,padding=1), # 32*64*64
            nn.ReLU(True),
            nn.MaxPool2d(2,2), # 32*32*32
            nn.BatchNorm2d(32),
        )
        self.flatten = nn.Flatten(start_dim=1)
        self.linear = nn.Sequential(
            nn.Linear(32 * 32 * 32, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128,64),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 32 * 32 * 32),
            nn.ReLU(True)
        )
        self.unflatten = nn.Unflatten(dim=1,
                                      unflattened_size=(32, 32, 32))
        self.cnn = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2,
                               padding=1, output_padding=1), #16*64*64
            nn.ReLU(True),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 16, 3, stride=1,
                               padding=1),  # 16*64*64
            nn.ReLU(True),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 8, 3, stride=2,
                               padding=1, output_padding=1), #8*128*128
            nn.ReLU(True),
            nn.BatchNorm2d(8),
            nn.ConvTranspose2d(8, 3, 3, stride=1,
                               padding=1),  #3*128*128
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.linear(x)
        x = self.unflatten(x)
        x = self.cnn(x)
        return x


class Linear_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.f=nn.Sequential(
            nn.Linear(64,128),
            nn.ReLU(True),
            nn.Linear(128,64),
        )

    def forward(self,x):
        return self.f(x)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=Encoder()
        self.decoder=Decoder()
        self.Linear_Model=Linear_Model()

    def forward(self,x):
        x1_old=x[:,0,:,:,:]
        x2_old=x[:,1,:,:,:]
        x1_code=self.encoder(x1_old)
        x2_code=self.encoder(x2_old)
        delta_x1=self.Linear_Model(x1_code)
        x1_new=self.decoder(x1_code)
        x2_new=self.decoder(x2_code)
        x1_next=self.decoder(x1_code+delta_x1)
        return x1_code,x2_code,delta_x1,x1_old,x1_new,x2_old,x2_new,x1_next


