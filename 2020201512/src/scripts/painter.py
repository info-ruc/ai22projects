from package import *
class HandTracker():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5, modelC=1):
        self.mode = mode  # 检测模式。设置为False则只在追踪手失败时才重新检测手；否则每一帧都重新检测手
        self.maxHands = maxHands  # 支持最大检测的手的个数。缺省则为2。
        self.detectionCon = detectionCon  # 0到1之间，缺省则为0.5。设置得越高则检测的手越准确，但是所用时间越长
        self.trackCon = trackCon  # 0到1之间，缺省则为0.5。设置得越高则越难追踪手。未追踪到需要重新检测手。
        self.modelC = modelC

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelC, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils  # 可视化。在图像或当前帧中查看结果

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 将图片转换成RGB三通道格式
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLm in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLm, self.mpHands.HAND_CONNECTIONS)
        return img

    def getPostion(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for lm in myHand.landmark:
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((cx, cy))

                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return lmList

    def getUpFingers(self, img):
        pos = self.getPostion(img, draw=False)
        self.upfingers = []
        if pos:
            # thumb
            self.upfingers.append((pos[4][1] < pos[3][1] and (pos[5][0] - pos[4][0] > 10)))
            # index
            self.upfingers.append((pos[8][1] < pos[7][1] and pos[7][1] < pos[6][1]))
            # middle
            self.upfingers.append((pos[12][1] < pos[11][1] and pos[11][1] < pos[10][1]))
            # ring
            self.upfingers.append((pos[16][1] < pos[15][1] and pos[15][1] < pos[14][1]))
            # pinky
            self.upfingers.append((pos[20][1] < pos[19][1] and pos[19][1] < pos[18][1]))
        return self.upfingers


class ColorRect():
    def __init__(self, x, y, w, h, color, text='', alpha=0.5):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.color = color
        self.text = text
        self.alpha = alpha

    def drawRect(self, img, text_color=(255, 255, 255), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, thickness=2):
        # draw the box
        alpha = self.alpha
        bg_rec = img[self.y: self.y + self.h, self.x: self.x + self.w]
        white_rect = np.ones(bg_rec.shape, dtype=np.uint8)
        white_rect[:] = self.color
        res = cv2.addWeighted(bg_rec, alpha, white_rect, 1 - alpha, 1.0)

        # Putting the image back to its position
        img[self.y: self.y + self.h, self.x: self.x + self.w] = res

        # put the letter
        tetx_size = cv2.getTextSize(self.text, fontFace, fontScale, thickness)
        text_pos = (int(self.x + self.w / 2 - tetx_size[0][0] / 2), int(self.y + self.h / 2 + tetx_size[0][1] / 2))
        cv2.putText(img, self.text, text_pos, fontFace, fontScale, text_color, thickness)

    def isOver(self, x, y):
        if (self.x + self.w > x > self.x) and (self.y + self.h > y > self.y):
            return True
        return False