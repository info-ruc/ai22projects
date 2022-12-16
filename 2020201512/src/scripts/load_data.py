from package import *
from utils import vector_2d_angle
resize_w = 640
resize_h = 640
# 初始化medialpipe
mp_drawing = mp.solutions.drawing_utils  # 作图工具
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands  # 手掌检测
hands = mp_hands.Hands(min_detection_confidence = 0.8, min_tracking_confidence = 0.8, max_num_hands = 1)
def read_image(directory_name):
    precessd_data = []
    k = 0
    for filename in os.listdir(directory_name):
        k = k + 1
        print("k=%d" % k)
        # if k > 3000:
        #     break
        image = cv2.imread(directory_name + "/" + filename)
        image = cv2.resize(image, (resize_w, resize_h))
        # 将图片格式设置成只读，提高图片格式转化的速度
        image.flags.writeable = False
        # 将图片格式转化为RGB格式
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 镜像处理
        image = cv2.flip(image, 1)
        # 将图像输入mediapipe模型，处理得到结果
        results = hands.process(image)
        flag = False
        if results.multi_hand_landmarks:
            # 遍历每个手掌
            for hand_landmarks in results.multi_hand_landmarks:
                # 解析手指，存入各个手指坐标
                hand_local = []
                # 遍历当前手的每个关节
                for i in range(21):
                    # 向handlocal中输入21个向量
                    x = hand_landmarks.landmark[i].x * image.shape[1]
                    y = hand_landmarks.landmark[i].y * image.shape[0]
                    hand_local.append((x, y))
                angle_ = vector_2d_angle(
                    ((int(hand_local[0][0]) - int(hand_local[5][0])), (int(hand_local[0][1]) - int(hand_local[5][1]))),
                    (0, 1)
                )
                n = int(angle_ / 90)
                n = n % 4
                image_copy = np.rot90(image, n)  # n=0,1,2,3,... 即旋转0，90,180,270，
                flag = True
                # cv2.imshow('image', image)
                # cv2.waitKey(300)
        if flag == False:
            continue
        results = hands.process(image_copy)
        # 将图片设置为可写状态并转回原来的格式
        image_copy.flags.writeable = True
        image_copy = cv2.cvtColor(image_copy, cv2.COLOR_RGB2BGR)
        # 判断是否有手掌
        # 当检测到手掌时，multi_hand_landmarks 不为空
        if results.multi_hand_landmarks:
            # 遍历每个手掌
            for hand_landmarks in results.multi_hand_landmarks:
                # 解析手指，存入各个手指坐标
                # 初始化一个列表来存储
                landmark_list = []
                # 遍历当前手的每个关节
                for landmark_id, finger_axis in enumerate(hand_landmarks.landmark):
                    # 向列表中存入手指序号，像素点的三维坐标
                    landmark_list.append([
                        finger_axis.x, finger_axis.y, finger_axis.z
                    ])

                # mp_drawing.draw_landmarks(
                #     # 图像
                #     image_copy,
                #     # 手指信息
                #     hand_landmarks,
                #     # 手指之间的连接
                #     mp_hands.HAND_CONNECTIONS,
                #     # 手指样式
                #     mp_drawing_styles.get_default_hand_landmarks_style(),
                #     # 连接样式
                #     mp_drawing_styles.get_default_hand_connections_style())
                # # 获取大拇指指尖坐标，序号为4
                # thumb_finger_tip = landmark_list[4]
                # # 向上取整，得到手指坐标的整数
                # # thumb_finger_tip[1]里存储的x值范围是0-1，乘以分辨率宽，便得到在图像上的位置
                # thumb_finger_tip_x = math.ceil(thumb_finger_tip[1] * resize_w)
                # # thumb_finger_tip[2]里存储的y值范围是0-1，乘以分辨率高，便得到在图像上的位置
                # thumb_finger_tip_y = math.ceil(thumb_finger_tip[2] * resize_h)
                # # 获取食指指尖坐标，序号为8，其他同上
                # index_finger_tip = landmark_list[8]
                # index_finger_tip_x = math.ceil(index_finger_tip[1] * resize_w)
                # index_finger_tip_y = math.ceil(index_finger_tip[2] * resize_h)
                # # 获取食指和拇指的中间点
                # finger_middle_point = (thumb_finger_tip_x + index_finger_tip_x) // 2, (
                #         thumb_finger_tip_y + index_finger_tip_y) // 2
                # # print(thumb_finger_tip_x)
                # thumb_finger_point = (thumb_finger_tip_x, thumb_finger_tip_y)
                # index_finger_point = (index_finger_tip_x, index_finger_tip_y)
                # # 用opencv的circle函数画图，将食指、拇指画出
                # image_copy = cv2.circle(image_copy, thumb_finger_point, 10, (255, 0, 255), -1)
                # image_copy = cv2.circle(image_copy, index_finger_point, 10, (255, 0, 255), -1)
                # cv2.imshow('image_copy', image_copy)
                # cv2.waitKey(300)

                precessd_data.append(landmark_list)
    return precessd_data

# 如果dataset路径改变可能需要更改下面的路径
call = read_image("D:\\pycharm\\pythonProject2\\dataset\\call")
dislike = read_image("D:\\pycharm\\pythonProject2\\dataset\\dislike")
fist = read_image("D:\\pycharm\\pythonProject2\\dataset\\fist")
ok = read_image("D:\\pycharm\\pythonProject2\\dataset\\ok")
one = read_image("D:\\pycharm\\pythonProject2\\dataset\\one")
palm = read_image("D:\\pycharm\\pythonProject2\\dataset\\palm")
peace = read_image("D:\\pycharm\\pythonProject2\\dataset\\peace")
stop = read_image("D:\\pycharm\\pythonProject2\\dataset\\stop")
mute = read_image("D:\\pycharm\\pythonProject2\\dataset\\mute")
three = read_image("D:\\pycharm\\pythonProject2\\dataset\\three")
three2 = read_image("D:\\pycharm\\pythonProject2\\dataset\\three2")
four = read_image("D:\\pycharm\\pythonProject2\\dataset\\four")
like = read_image("D:\\pycharm\\pythonProject2\\dataset\\like")
two_up = read_image("D:\\pycharm\\pythonProject2\\dataset\\two_up")
data = []
data.append(call)
data.append(dislike)
data.append(fist)
data.append(ok)
data.append(one)
data.append(palm)
data.append(peace)
data.append(stop)
data.append(mute)
data.append(three)
data.append(three2)
data.append(four)
data.append(like)
data.append(two_up)
with h5py.File('train_signs.h5', 'w') as f:
    train_set_x = f.create_dataset('train_set_x', [0, 21, 3], maxshape=[None, 21, 3])
    train_set_y = f.create_dataset('train_set_y', [0, 1], maxshape=[None, 1])
    test_set_x = f.create_dataset('test_set_x', [0, 21, 3], maxshape=[None, 21, 3])
    test_set_y = f.create_dataset('test_set_y', [0, 1], maxshape=[None, 1])
    i = 0
    for _data in data:
        j = 0
        for single_data in _data:
            print("i=%d" % i)
            if j < 1000:
                temp = test_set_x.shape[0]
                print("temp=%d"%temp)
                test_set_x.resize((temp + 1, 21, 3))
                test_set_x[temp:temp + 1] = single_data
                test_set_y.resize((temp + 1, 1))
                test_set_y[temp:temp + 1] = int(i)
                print(test_set_y)
                print(test_set_y[:])
            else:
                temp = train_set_x.shape[0]
                train_set_x.resize((temp + 1, 21, 3))
                train_set_x[temp:temp + 1] = single_data
                train_set_y.resize((temp + 1, 1))
                train_set_y[temp:temp + 1] = int(i)
            j = j + 1
        i = i + 1