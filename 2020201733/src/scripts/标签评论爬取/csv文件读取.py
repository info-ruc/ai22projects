#已爬取3005本
import requests
import re
import bs4
import lxml
import pandas as pd
import time
import random


data = pd.read_csv("BookStacks_all.csv",encoding = 'utf-8')
i = 6000
print(data.loc[i])
# print(data[1])
print(data.loc[i,'链接'])
print(data.loc[i,'评论'])
print(len(data))

# data_2 = pd.read_csv("BookStacks2.csv",encoding='utf-8')
# for j in range(82,1000):
#     data_2.loc[j] = data.loc[j]
#     data.to_csv("BookStacks2.csv", sep=',', index=False, header=True)