import requests
import re
import bs4
import lxml
import pandas as pd
import time
import random

k = 0

def get_info(url):
    '''打开文件'''
    data = pd.read_csv("BookStacks2.csv")
    '''抓取网站html'''
    headers = {
        'User-Agent': 'my-app/0.0.1'}
    r = requests.get(url, headers=headers)
    r = r.text

    # print(r)
    html = r
    soup = bs4.BeautifulSoup(html,'html.parser')
    '''获取链接'''
    link = soup.find_all("a",class_ = "book-name")
    link_len = len(link)
    '''获取字数，评分，状态'''
    nums = soup.find_all(text=re.compile("[0-9][w]*万?亿?字"))
    # num_len = len(nums)
    scores = soup.find_all(text=re.compile("综合评分"))
    years = soup.find_all(text=re.compile("完结|连载|太监"))
    # year_len = len(years)
    score_len = len(scores)
    print(score_len)
    for i in range(0, score_len):
        scores[i] = scores[i].strip()
    global k
    start = time.perf_counter()
    for i in range(0,link_len):
        link_get = link[i].get('href')
        '''获取书单名'''
        name = link[i].text
        data.loc[k+i,"书名"] = name
        '''可跳转地址'''
        link_t = "https://www.yousuu.com"+ link_get
        data.loc[k + i, "链接"] = link_t
        '''字数'''
        num = nums[i]
        data.loc[k + i, "字数"] = num
        '''浏览量'''
        year = years[i]
        data.loc[k + i, "完结状态"] = year
        # '''收藏量'''
        score = scores[i]
        data.loc[k + i, "评分"] = score
        progress = i + 1
        a = "*" * progress
        b = "." * (link_len - progress)
        c = (progress / link_len) * 100
        dur = time.perf_counter() - start
        print("\r{:^3.0f}%[{}->{}]{:.2f}s".format(c, a, b, dur), end="")

    '''更新全局变量'''
    k += link_len
    data.to_csv("BookStacks2.csv", sep=',', index=False, header=True)
    # print(link_t)
    # print(soup.title)

    '''获取书数，浏览量，收藏量'''
    # num = soup.find_all('span',class_ = "ResultBooklistIteMeta")
    # book_num = soup.find_all(text=re.compile("本书"))
    # views = soup.find_all(text=re.compile("浏览"))
    # collect = soup.find_all(text=re.compile("收藏"))
    # print(book_num,views,collects)
    '''进度条'''
    # start = time.perf_counter()
    #
    # for i in range(scale + 1):
    #
    # a = "*" * i
    #
    # b = "." * (scale - i)
    #
    # c = (i / scale) * 100
    #
    # dur = time.perf_counter() - start
    #
    # print("\r{:^3.0f}%[{}->{}]{:.2f}s".format(c, a, b, dur), end="")


def main():
    headers = ['书名', '字数', '完结状态', '评分','链接']
    with open('BookStacks2.csv', 'w', encoding='utf8', newline='') as f:
        f.writelines(headers)
    # page = 1
    for page in range(280,501):
        print("正在爬取第{}页书籍中……".format(page))
        '''man'''
        # url = 'https://www.yousuu.com/booklists/?type=man&screen=comprehensive&page=' + page.__str__()
        '''woman'''
        # https: // www.yousuu.com / booklists /?type = woman & screen = comprehensive & page = 1
        url = 'https://www.yousuu.com/bookstore/?channel&classId&tag&countWord&status&update&sort&page=' + page.__str__()
        get_info(url)
        print("第{}页书籍爬取完毕！".format(page))
        print("休息中……")
        '''随机访问间隔防止反爬'''
        time.sleep(random.randint(1,10))
        # data = pd.read_csv('书单.csv')

    print("已完成爬取任务！！")


main()

# headers = {
#     'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.60 Safari/537.36 Edg/100.0.1185.29'}
# resq = requests.get(url=url)
# print(resq)
# # print(resq.text)
# text = resq.text
# # pattern = '<div.*?<main.*?<ol.*?<li.*?href=("http.*?/")'
# # html = re.findall(pattern,text,re.S)
# # for i in html:
# #     print(i)
# bs = bs4.BeautifulSoup(text,"lxml")
# print(bs)