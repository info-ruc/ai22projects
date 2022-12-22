#import argparse
from crnn import CRNN

# 模型训练
def train():

# 设置基本属性
    batch_size=32    # 批量大小
    max_image_width=400   # 最大图片宽度
    train_test_ratio=0.75    # 训练集、测试集划分比例
    restore=True    # 是否恢复加载模型，可用于多次加载训练
    iteration_count=100    # 迭代次数
    # 初始化调用CRNN
    crnn = CRNN(
        batch_size,
        'E:\人工智能导论\CRNN-master\CRNN\model',
        'E:\人工智能导论\CRNN-master\CRNN\data',
        max_image_width,
        train_test_ratio,
        restore,
        "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-'.!?,\"&",
        False,
        'en'
    )
    # 模型训练
    crnn.train(iteration_count)
    
#模型测试
# 模型测试
def test():

# 设置基本属性
    batch_size=32
    max_image_width=400
    restore=True
    # 初始化CRNN
    crnn = CRNN(
        batch_size,
        r'E:\人工智能导论\CRNN-master\CRNN\model',
        r'E:\人工智能导论\CRNN-master\CRNN\test_data',
        max_image_width,
        0,
        restore,
        "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-'.!?,\"&",
        False,
        'en'
    )
    # 测试模型
    crnn.test()

test()

