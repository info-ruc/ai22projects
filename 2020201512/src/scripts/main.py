from package import *
from predict import predict_MLP
from utils import h_gesture
from utils import translate
from utils import hand_angle
from utils import vector_2d_angle
from utils import cal_vector_abs
from painter import *
#initilize the habe detector
detector = HandTracker()
choice = 0
def on_release(key):
    print('{0} released'.format(key))
    if key == keyboard.Key.tab:
        global choice
        choice += 1
        choice %= 5
        print("choice = %d" %choice)

tf.compat.v1.disable_eager_execution()
parameters = pickle.load(open("parameters", 'rb'))
# 初始化medialpipe
mp_drawing = mp.solutions.drawing_utils  # 作图工具
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands  # 手掌检测
hands = mp_hands.Hands(min_detection_confidence = 0.8, min_tracking_confidence = 0.8, max_num_hands = 1)

# 获取电脑音量范围
devices = AudioUtilities.GetSpeakers()  # 初始化windows音频控制对象
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)  # 调用系统音频控制接口
volume = cast(interface, POINTER(IAudioEndpointVolume))
volume.SetMute(0, None)  # 音量归零
volume_range = volume.GetVolumeRange()  # 获取电脑音量范围


# 计算刷新率
fpsTime = time.time()
# 调用time函数记下程序运行到此处的时间，秒为单位用于计算刷新率

# OpenCV读取视频流
cap = cv2.VideoCapture(0)
# 初始化一个OpenCV读取视频流的对象，用于通过摄像头获取视频

# 视频分辨率
resize_w = 1280
resize_h = 720
cap.set(3, resize_w)
cap.set(4, resize_h)

# creating canvas to draw on it
canvas = np.zeros((resize_h, resize_w, 3), np.uint8)

# define a previous point to be used with drawing a line
px,py = 0,0
#initial brush color
color = (255,0,0)
#####
brushSize = 5
eraserSize = 20
####

########### creating colors ########
# Colors button
colorsBtn = ColorRect(200, 0, 100, 100, (120,255,0), 'Colors')
colors = []
#random color
b = int(random.random()*255)-1
g = int(random.random()*255)
r = int(random.random()*255)
print(b,g,r)
colors.append(ColorRect(300,0,100,100, (b,g,r)))
#red
colors.append(ColorRect(400,0,100,100, (0,0,255)))
#blue
colors.append(ColorRect(500,0,100,100, (255,0,0)))
#green
colors.append(ColorRect(600,0,100,100, (0,255,0)))
#yellow
colors.append(ColorRect(700,0,100,100, (0,255,255)))
#erase (black)
colors.append(ColorRect(800,0,100,100, (0,0,0), "Eraser"))

#clear
clear = ColorRect(900,0,100,100, (100,100,100), "Clear")

########## pen sizes #######
pens = []
for i, penSize in enumerate(range(5,25,5)):
    pens.append(ColorRect(1100,50+100*i,100,100, (50,50,50), str(penSize)))

penBtn = ColorRect(1100, 0, 100, 50, color, 'Pen')

# white board button
boardBtn = ColorRect(50, 0, 100, 100, (255,255,0), 'Board')

#define a white board to draw on
whiteBoard = ColorRect(50, 120, 1020, 580, (255,255,255),alpha = 0.6)

coolingCounter = 20
hideBoard = True
hideColors = True
hidePenSizes = True

# 画面显示初始化参数
rect_height = 0
rect_percent_text = 0

keyboard.Listener(on_release=on_release).start()
# 调用mediapipe的Hands函数，输入手指关节检测的置信度和上一帧跟踪的置信度，输入最多检测手的数目，进行关节点检测

# 摄像头打开则程序一直运行
while cap.isOpened():
    if coolingCounter:
        coolingCounter -=1
    success, image = cap.read()
    # 获取一帧当前图像，返回是否获取成功和图像数组（用numpy矩阵存储的照片）

    image = cv2.resize(image, (resize_w, resize_h))
    # 修改图像的大小为设置的分辨率大小

    # 如果没有成功获取当前帧的图像则获取下一帧图像
    if not success:
        print("空帧.")
        continue

    # 将图片格式设置成只读，提高图片格式转化的速度
    image.flags.writeable = False
    # 将图片格式转化为RGB格式
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 镜像处理
    image = cv2.flip(image, 1)
    # 将图像输入mediapipe模型，处理得到结果
    results = hands.process(image)
    # 将图片设置为可写状态并转回原来的格式
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    detector.findHands(image)
    positions = detector.getPostion(image, draw=False)
    upFingers = detector.getUpFingers(image)
    if choice == 4:
        if upFingers:
            x, y = positions[8][0], positions[8][1]
            if upFingers[1] and not whiteBoard.isOver(x, y):
                px, py = 0, 0

                ##### pen sizes ######
                if not hidePenSizes:
                    for pen in pens:
                        if pen.isOver(x, y):
                            brushSize = int(pen.text)
                            pen.alpha = 0
                        else:
                            pen.alpha = 0.5

                ####### chose a color for drawing #######
                if not hideColors:
                    for cb in colors:
                        if cb.isOver(x, y):
                            color = cb.color
                            cb.alpha = 0
                        else:
                            cb.alpha = 0.5

                    # Clear
                    if clear.isOver(x, y):
                        clear.alpha = 0
                        canvas = np.zeros((720, 1280, 3), np.uint8)
                    else:
                        clear.alpha = 0.5

                # color button
                if colorsBtn.isOver(x, y) and not coolingCounter:
                    coolingCounter = 10
                    colorsBtn.alpha = 0
                    hideColors = False if hideColors else True
                    colorsBtn.text = 'Colors' if hideColors else 'Hide'
                else:
                    colorsBtn.alpha = 0.5

                # Pen size button
                if penBtn.isOver(x, y) and not coolingCounter:
                    coolingCounter = 10
                    penBtn.alpha = 0
                    hidePenSizes = False if hidePenSizes else True
                    penBtn.text = 'Pen' if hidePenSizes else 'Hide'
                else:
                    penBtn.alpha = 0.5

                # white board button
                if boardBtn.isOver(x, y) and not coolingCounter:
                    coolingCounter = 10
                    boardBtn.alpha = 0
                    hideBoard = False if hideBoard else True
                    boardBtn.text = 'Board' if hideBoard else 'Hide'

                else:
                    boardBtn.alpha = 0.5




            elif upFingers[1] and not upFingers[2]:
                if whiteBoard.isOver(x, y) and not hideBoard:
                    # print('index finger is up')
                    cv2.circle(image, positions[8], brushSize, color, -1)
                    # drawing on the canvas
                    if px == 0 and py == 0:
                        px, py = positions[8]
                    if color == (0, 0, 0):
                        cv2.line(canvas, (px, py), positions[8], color, eraserSize)
                    else:
                        cv2.line(canvas, (px, py), positions[8], color, brushSize)
                    px, py = positions[8]

            else:
                px, py = 0, 0

        # put colors button
        colorsBtn.drawRect(image)
        cv2.rectangle(image, (colorsBtn.x, colorsBtn.y), (colorsBtn.x + colorsBtn.w, colorsBtn.y + colorsBtn.h),
                      (255, 255, 255), 2)

        # put white board buttin
        boardBtn.drawRect(image)
        cv2.rectangle(image, (boardBtn.x, boardBtn.y), (boardBtn.x + boardBtn.w, boardBtn.y + boardBtn.h),
                      (255, 255, 255), 2)

        # put the white board on the frame
        if not hideBoard:
            whiteBoard.drawRect(image)
            ########### moving the draw to the main image #########
            canvasGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            _, imgInv = cv2.threshold(canvasGray, 20, 255, cv2.THRESH_BINARY_INV)
            imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
            image = cv2.bitwise_and(image, imgInv)
            image = cv2.bitwise_or(image, canvas)

        ########## pen colors' boxes #########
        if not hideColors:
            for c in colors:
                c.drawRect(image)
                cv2.rectangle(image, (c.x, c.y), (c.x + c.w, c.y + c.h), (255, 255, 255), 2)

            clear.drawRect(image)
            cv2.rectangle(image, (clear.x, clear.y), (clear.x + clear.w, clear.y + clear.h), (255, 255, 255), 2)

        ########## brush size boxes ######
        penBtn.color = color
        penBtn.drawRect(image)
        cv2.rectangle(image, (penBtn.x, penBtn.y), (penBtn.x + penBtn.w, penBtn.y + penBtn.h), (255, 255, 255), 2)
        if not hidePenSizes:
            for pen in pens:
                pen.drawRect(image)
                cv2.rectangle(image, (pen.x, pen.y), (pen.x + pen.w, pen.y + pen.h), (255, 255, 255), 2)

    # 判断是否有手掌
    # 当检测到手掌时，multi_hand_landmarks 不为空
    if results.multi_hand_landmarks:
        # 遍历每个手掌
        for hand_landmarks in results.multi_hand_landmarks:
            # 在画面标注手指
            mp_drawing.draw_landmarks(
                # 图像
                image,
                # 手指信息
                hand_landmarks,
                # 手指之间的连接
                mp_hands.HAND_CONNECTIONS,
                # 手指样式
                mp_drawing_styles.get_default_hand_landmarks_style(),
                # 连接样式
                mp_drawing_styles.get_default_hand_connections_style())

            # 解析手指，存入各个手指坐标
            # 初始化一个列表来存储
            landmark_list = []
            hand_local = []
            X_test_orig = []
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
            for landmark_id, finger_axis in enumerate(hand_landmarks.landmark):
                # 向列表中存入手指序号，像素点的三维坐标
                landmark_list.append([
                    landmark_id, finger_axis.x, finger_axis.y,
                    finger_axis.z
                ])
                temp_x = finger_axis.x;
                temp_y = finger_axis.y;
                while n > 0:
                    temp = temp_x
                    temp_x = temp_y
                    temp_y = 1 - temp
                    n = n - 1
                X_test_orig.append([
                    temp_x, temp_y, finger_axis.z
                ])
            if landmark_list:
                # 获取大拇指指尖坐标，序号为4
                thumb_finger_tip = landmark_list[4]
                # 向上取整，得到手指坐标的整数
                # thumb_finger_tip[1]里存储的x值范围是0-1，乘以分辨率宽，便得到在图像上的位置
                thumb_finger_tip_x = math.ceil(thumb_finger_tip[1] * resize_w)
                # thumb_finger_tip[2]里存储的y值范围是0-1，乘以分辨率高，便得到在图像上的位置
                thumb_finger_tip_y = math.ceil(thumb_finger_tip[2] * resize_h)

                # 获取食指指尖坐标，序号为8，其他同上
                index_finger_tip = landmark_list[8]
                index_finger_tip_x = math.ceil(index_finger_tip[1] * resize_w)
                index_finger_tip_y = math.ceil(index_finger_tip[2] * resize_h)

                # 获取食指和拇指的中间点
                finger_middle_point = (thumb_finger_tip_x + index_finger_tip_x) // 2, (
                        thumb_finger_tip_y + index_finger_tip_y) // 2
                # print(thumb_finger_tip_x)
                thumb_finger_point = (thumb_finger_tip_x, thumb_finger_tip_y)
                index_finger_point = (index_finger_tip_x, index_finger_tip_y)

                # 用opencv的circle函数画图，将食指、拇指画出
                image = cv2.circle(image, thumb_finger_point, 10, (255, 0, 255), -1)
                image = cv2.circle(image, index_finger_point, 10, (255, 0, 255), -1)

                if choice == 0:
                    angle_list = hand_angle(hand_local)
                    gesture_str = h_gesture(angle_list)
                    cv2.putText(image, gesture_str, (10, 110), 0, 1.3, (0, 0, 255), 3)

                if choice== 3:
                    X_test_orig = np.array(X_test_orig)
                    X_test_orig = X_test_orig.reshape(1, -1).T
                    y_pred = predict_MLP(X_test_orig, parameters)
                    str_tmp = translate(y_pred[0])
                    cv2.putText(image, str_tmp, (10, 110), 0, 1.3, (0, 0, 255), 3)

                if choice == 1:
                    # 用opencv的circle函数画图，将食指、拇指中间点画出
                    image = cv2.circle(image, finger_middle_point, 10, (255, 0, 255), -1)
                    # 用opencv的line函数将食指和拇指连接在一起
                    image = cv2.line(image, thumb_finger_point, index_finger_point, (255, 0, 255), 5)
                    # 勾股定理计算长度。math.hypot为勾股定理计算两点长度的函数，得到食指和拇指的距离
                    line_len = math.hypot((index_finger_tip_x - thumb_finger_tip_x),
                                          (index_finger_tip_y - thumb_finger_tip_y))
                    # 获取电脑最大最小音量
                    min_volume = volume_range[0]
                    max_volume = volume_range[1]
                    # 将指尖长度映射到音量上
                    # np.interp为插值函数，简而言之，看line_len的值在[50，200]中所占比例，然后去[min_volume,max_volume]中线性寻找相应的值，作为返回值
                    temp = cal_vector_abs((int(hand_local[0][0]) - int(hand_local[5][0])), (int(hand_local[0][1]) - int(hand_local[5][1])))
                    factor = temp / 200

                    vol = np.interp(line_len, [50 * factor, 200 * factor], [min_volume, max_volume])
                    # 将指尖长度映射到矩形显示上
                    # 同理，通过line_len与[50，200]的比较，得到音量百分比
                    rect_height = np.interp(line_len, [50 * factor, 200 * factor], [0, 200])
                    rect_percent_text = np.interp(line_len, [50 * factor, 200 * factor], [0, 100])
                    # 用之前得到的vol设置电脑音量
                    volume.SetMasterVolumeLevel(vol, None)

                if choice == 2:
                    # 用opencv的circle函数画图，将食指、拇指中间点画出
                    image = cv2.circle(image, finger_middle_point, 10, (255, 0, 255), -1)
                    # 用opencv的line函数将食指和拇指连接在一起
                    image = cv2.line(image, thumb_finger_point, index_finger_point, (255, 0, 255), 5)
                    # 勾股定理计算长度。math.hypot为勾股定理计算两点长度的函数，得到食指和拇指的距离
                    line_len = math.hypot((index_finger_tip_x - thumb_finger_tip_x),
                                          (index_finger_tip_y - thumb_finger_tip_y))
                    temp = cal_vector_abs((int(hand_local[0][0]) - int(hand_local[5][0])), (int(hand_local[0][1]) - int(hand_local[5][1])))
                    factor = temp / 200
                    # 将指尖长度映射到亮度上
                    # np.interp为插值函数，简而言之，看line_len的值在[50，200]中所占比例，然后去[0,100]中线性寻找相应的值，作为返回值
                    bri = np.interp(line_len, [50 * factor, 200 * factor], [0, 100])
                    # 将指尖长度映射到矩形显示上
                    # 同理，通过line_len与[50，200]的比较，得到亮度百分比
                    rect_height = np.interp(line_len, [50 * factor, 200 * factor], [0, 200])
                    rect_percent_text = np.interp(line_len, [50 * factor, 200 * factor], [0, 100])
                    # 用之前得到的bri设置电脑音量
                    sbc.set_brightness(bri)

    if choice == 1:
        # 通过opencv的putText函数，将音量百分比显示到图像上
        cv2.putText(image, str(math.ceil(rect_percent_text)) + "%", (10, 350), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3)
        # 通过opencv的rectangle函数，画出透明矩形框
        image = cv2.rectangle(image, (30, 100), (70, 300), (255, 0, 0), 3)
        image = cv2.rectangle(image, (30, math.ceil(300 - rect_height)), (70, 300), (255, 0, 0), -1)

    if choice == 2:
        # 通过opencv的putText函数，将音量百分比显示到图像上
        cv2.putText(image, str(math.ceil(rect_percent_text)) + "%", (10, 350), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3)
        # 通过opencv的rectangle函数，画出透明矩形框
        image = cv2.rectangle(image, (30, 100), (70, 300), (255, 0, 0), 3)
        image = cv2.rectangle(image, (30, math.ceil(300 - rect_height)), (70, 300), (255, 0, 0), -1)

    # 显示刷新率FPS，cTime为程序一个循环截至的时间
    cTime = time.time()
    fps_text = 1 / (cTime - fpsTime)  # 计算频率
    fpsTime = cTime  # 下一轮开始的时间置为这一轮循环结束的时间
    # 显示帧率
    cv2.putText(image, "FPS: " + str(int(fps_text)), (10, 40),cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    # 显示当前模式
    if choice == 0:
        cv2.putText(image, "mode: gesture-identify", (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)
    if choice == 1:
        cv2.putText(image, "mode: hand-control-volume", (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)
    if choice == 2:
        cv2.putText(image, "mode: hand-control-brightness", (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)
    if choice == 3:
        cv2.putText(image, "mode: gesture-identify2", (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)
    if choice == 4:
        cv2.putText(image, "mode: paint", (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)
    # 用opencv的函数显示摄像头捕捉的画面，以及在画面上写的字，画的框
    cv2.imshow('MediaPipe Hands', image)
    # 每次循环等待5毫秒，如果按下Esc或者窗口退出，这跳出循环
    if cv2.waitKey(1) & 0xFF == 27 or cv2.getWindowProperty('MediaPipe Hands', cv2.WND_PROP_VISIBLE) < 1:
        break
# 释放对视频流的获取
cap.release()