#已爬取9999本
import requests
import re
import bs4
import lxml
import pandas as pd
import time
import random
security = 0
'''设置cookie以查看评论'''
cookie = "__gads=ID=2f7ca3edf11be2d2-2249cf98a1d80015:T=1669249349:RT=1669249349:S=ALNI_MbqYj_LfqIx79dlMfg5GzVjh_q0gQ; Hm_lvt_42e120beff2c918501a12c0d39a4e067=1669207867,1669248838,1669272992,1669336984; __gpi=UID=00000b82338e9a56:T=1669249349:RT=1669336984:S=ALNI_MbqDkOpjyjI6yz_Cp2JjQaRgGvxOQ; token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6MTc2ODg5OCwiaWF0IjoxNjY5MzM3MTQxLCJleHAiOjE2NzE5MjkxNDF9.05Ce-Bfo1C1m8_Go3MXfYdHPnKn4b1fivbNBXHFlcO0; token.sig=HOAN-qT9C3RFijBYc4Z0RTMEnB7ZRK6wKOyDyUfYuHM; Hm_lpvt_42e120beff2c918501a12c0d39a4e067=1669337149"
headers = {
    'User-Agent': 'my-app/0.0.1',
    'Cookie':cookie}

data = pd.read_csv("BookStacks_all.csv",encoding = 'utf-8')
# # data_new=data.drop(["书名字数完结状态评分链接  "],axis=1) #删除title这列数据
# # print(data_new.loc[1])
# print(data.loc[1])
# # print(data[1])
# print(data.loc[1,'链接'])
# print(len(data))
length = len(data)
start = time.perf_counter()
for i in range(8002,length):
    url = data.loc[i,'链接']
    r = requests.get(url, headers=headers)
    r = r.text
    html = r
    soup = bs4.BeautifulSoup(html, 'html.parser')
    title_host = soup.find_all('span', class_='el-tag el-tag--primary el-tag--small el-tag--light')
    host_len = len(title_host)
    '''主标签'''
    print("正在打标签中，请稍等")
    print("正在写评论中，请稍等")
    for k in range(0, host_len):
        data.loc[i, "主标签"] = title_host[k].text
        # print(title_host[k].text)

        title_sec = soup.find_all('span', class_='el-tag el-tag--success el-tag--small el-tag--light')
        sec_len = len(title_sec)
        '''副标签'''
        if sec_len == 0:
            data.loc[i, "副标签"] = 'NULL'
            # print('NULL')
        elif sec_len == 1:
            data.loc[i, '副标签'] = title_sec[0].text
        else:
            # 由于副标签可能有多个值，故需以字符串形式连接存放在一个单元格
            title_p = title_sec[0].text
            for j in range(1, sec_len):
                title_p = title_p + ',' + title_sec[j].text
            data.loc[i, '副标签'] = title_p
            # print(title_sec[j].text)
        '''进度条'''
    '''爬评论'''
    comments = soup.find_all(text=re.compile("\"[\u4e00-\u9fa5]+\""))
    comments_str = str(comments)
    '''保留中文以及标点'''
    t = re.findall('[\u3002\uff1b\uff0c\uff1a\u201c\u201d\uff08\uff09\u3001\uff1f\u300a\u300b\u4e00-\u9fa5]',
                   comments_str)
    t = ''.join(t)
    t_str = str(t)
    t_cut = t_str.split('。')
    '''去除最后一个无意义'''
    t_cut.pop()
    t_cut = str(t_cut)
    data.loc[i, '评论'] = t_cut
    progress = i + 1
    a = "*" * progress
    b = "." * (length - progress)
    c = (progress / length) * 100
    dur = time.perf_counter() - start
    print("\r{:^3.0f}%[{}->{}]{:.2f}s".format(c, a, b, dur), end="")
    print("标签获取成功")
    print("评论写成功")
    print("第{}本书籍爬取完毕！".format(i))
    data.to_csv("BookStacks_all.csv", sep=',', index=False, header=True)
    print("第{}本书籍保存完毕！".format(i))
    print("休息中……")
    # '''随机访问间隔防止反爬'''
    # time.sleep(random.randint(1, 10))
'''保存'''
data.to_csv("BookStacks_all.csv", sep=',', index=False, header=True)
print("信息提取完毕")