import requests
import os
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.89 Safari/537.36 SLBrowser/7.0.0.10251 SLBChan/105"
    }


def loadImg(index, maxnum=100):
    loadnum = 0
    while loadnum < maxnum:
        url = "https://image.baidu.com/search/acjson?tn=resultjson_com&ipn=rj&ct=201326592&is=0%2C0&"\
              "fp=detail&logid=11962624943566928039&cl=2&lm=-1&ie=utf-8&oe=utf-8&adpicid=0&lpn=0&st=-1&word="+index+"&z=0&ic=0&hd=undefined&latest=undefined&copyright=undefined&s=undefined&se=&tab=0&width=&height=&face=undefined&istype=2&qc=&nc=&fr=&simics=&srctype=&bdtype=0&rpstart=0&rpnum=0&cs=3464307413%2C312436372"\
              "&catename=&nojc=undefined&album_id=&album_tab=&cardserver=&tabname=&pn="+str(loadnum)+"&rn=30&gsm=4&1638875927992="
        response = requests.get(url, headers=headers, timeout=3)
        # 请求状态
        if response.status_code == 200:
            print("请求成功！")
        # json文件-data-【0-30】-hoverurl
        for i in range(30):
            ImgUrl = response.json(strict=False)['data'][i]["hoverURL"]
            Img = requests.get(url=ImgUrl, headers=headers)
            name = loadnum + i
            # 进度显示
            print(name)
            # 二进制保存图片
            with open(save_path+index+str(name)+".jpg", "wb") as f:
                f.write(Img.content)
        loadnum += 30


def loadImgtest(index, maxnum=20):
    loadnum = 0
    while loadnum < maxnum:
        url = "https://image.baidu.com/search/acjson?tn=resultjson_com&ipn=rj&ct=201326592&is=0%2C0&"\
              "fp=detail&logid=11962624943566928039&cl=2&lm=-1&ie=utf-8&oe=utf-8&adpicid=0&lpn=0&st=-1&word="+index+"&z=0&ic=0&hd=undefined&latest=undefined&copyright=undefined&s=undefined&se=&tab=0&width=&height=&face=undefined&istype=2&qc=&nc=&fr=&simics=&srctype=&bdtype=0&rpstart=0&rpnum=0&cs=3464307413%2C312436372"\
              "&catename=&nojc=undefined&album_id=&album_tab=&cardserver=&tabname=&pn="+str(loadnum)+"&rn=30&gsm=4&1638875927992="
        response = requests.get(url, headers=headers, timeout=3)
        # 请求状态
        if response.status_code == 200:
            print("请求成功！")
        # json文件-data-【0-30】-hoverurl
        for i in range(20):
            ImgUrl = response.json(strict=False)['data'][i]["hoverURL"]
            Img = requests.get(url=ImgUrl, headers=headers)
            name = loadnum + i
            # 进度显示
            print(name)
            # 二进制保存图片
            with open(save_path+index+str(name)+".jpg", "wb") as f:
                f.write(Img.content)
        loadnum += 20


# 同类别索引
labels = ["鸟", "狗狗"]
index = {"鸟": ['苍鹰', '麻雀', '猫头鹰', '鸽子', '燕子', '喜鹊', '翠鸟', '乌鸦'],
         "狗狗": ['萨摩耶', '拉布拉多', '柯基', '金毛', '秋田犬', '法斗', '捷克狼犬', '边牧']}
need_num = [800, 800]  # 需要的图片数量
test_num = [150, 150]  # 需要的图片数量


def mkdir(path):
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
        print
        path + ' 创建成功'
        return True
    else:
        return True


for num in range(0, len(labels)):
    # 需要提前创建label文件夹
    times = need_num[num]//150
    mkdir(r"C:/Users/60553/Desktop/rgzn/czxt/"+labels[num])
    save_path = r"C:/Users/60553/Desktop/rgzn/czxt/"+labels[num]+"/"
    for i in range(times):
        loadImg(index[labels[num]][i])

for num in range(0, len(labels)):
    # 需要提前创建label文件夹
    times = test_num[num]//20
    mkdir(r"C:/Users/60553/Desktop/rgzn/testdata/"+labels[num])
    save_path = r"C:/Users/60553/Desktop/rgzn/testdata/"+labels[num]+"/"
    for i in range(times):
        loadImgtest(index[labels[num]][i])