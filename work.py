import random
import cv2
from mediapipe import solutions

# 定义一个函数来识别手势
def gesture_recognizer(hand_landmarks):
    # 使用的一些关键点参数
    p0_x = hand_landmarks.landmark[0].x
    p0_y = hand_landmarks.landmark[0].y

    p1_x = hand_landmarks.landmark[1].x
    p1_y = hand_landmarks.landmark[1].y

    p4_x = hand_landmarks.landmark[4].x
    p4_y = hand_landmarks.landmark[4].y

    p5_x = hand_landmarks.landmark[5].x
    p5_y = hand_landmarks.landmark[5].y

    p8_x = hand_landmarks.landmark[8].x
    p8_y = hand_landmarks.landmark[8].y

    p9_x = hand_landmarks.landmark[9].x
    p9_y = hand_landmarks.landmark[9].y

    p12_x = hand_landmarks.landmark[12].x
    p12_y = hand_landmarks.landmark[12].y

    p13_x = hand_landmarks.landmark[13].x
    p13_y = hand_landmarks.landmark[13].y

    p16_x = hand_landmarks.landmark[16].x
    p16_y = hand_landmarks.landmark[16].y

    p17_x = hand_landmarks.landmark[17].x
    p17_y = hand_landmarks.landmark[17].y

    p20_x = hand_landmarks.landmark[20].x
    p20_y = hand_landmarks.landmark[20].y

    distance0_1 = ((p0_x - p1_x) ** 2 + (p0_y - p1_y) ** 2) ** 0.5
    distance0_4 = ((p0_x - p4_x) ** 2 + (p0_y - p4_y) ** 2) ** 0.5
    distance1_4 = ((p1_x - p4_x) ** 2 + (p1_y - p4_y) ** 2) ** 0.5
    distance0_5 = ((p0_x - p5_x) ** 2 + (p0_y - p5_y) ** 2) ** 0.5
    distance0_8 = ((p0_x - p8_x) ** 2 + (p0_y - p8_y) ** 2) ** 0.5
    distance0_9 = ((p0_x - p9_x) ** 2 + (p0_y - p9_y) ** 2) ** 0.5
    distance0_12 = ((p0_x - p12_x) ** 2 + (p0_y - p12_y) ** 2) ** 0.5
    distance0_13 = ((p0_x - p13_x) ** 2 + (p0_y - p13_y) ** 2) ** 0.5
    distance0_16 = ((p0_x - p16_x) ** 2 + (p0_y - p16_y) ** 2) ** 0.5
    distance0_17 = ((p0_x - p17_x) ** 2 + (p0_y - p17_y) ** 2) ** 0.5
    distance0_20 = ((p0_x - p20_x) ** 2 + (p0_y - p20_y) ** 2) ** 0.5



    # if (distance0_1 ** 2 + distance1_4 ** 2 -distance0_4 ** 2)/(2 * distance0_1 * distance1_4)<-0.7  #拇指伸开
    # if -0.7<(distance0_1 ** 2 + distance1_4 ** 2 -distance0_4 ** 2)/(2 * distance0_1 * distance1_4)<-0.5  #拇指与手掌距离较近
    # if (distance0_5/distance0_8)<0.8  #食指伸开
    # if (distance0_9/distance0_12)<0.8  #中指伸开
    # if (distance0_13/distance0_16)<0.8  #无名指伸开
    # if (distance0_17/distance0_20)<0.8  #小拇指伸开

    if (distance0_1 ** 2 + distance1_4 ** 2 -distance0_4 ** 2)/(2 * distance0_1 * distance1_4)>=-0.7 and (distance0_5/distance0_8)>=0.8 and (distance0_9/distance0_12)>=0.8 and (distance0_13/distance0_16)>=0.8 and (distance0_17/distance0_20)>=0.8:
        return 1
    elif (distance0_1 ** 2 + distance1_4 ** 2 -distance0_4 ** 2)/(2 * distance0_1 * distance1_4)<-0.5 and (distance0_5/distance0_8)<0.8 and (distance0_9/distance0_12)<0.8 and (distance0_13/distance0_16)<0.8 and (distance0_17/distance0_20)<0.8:
        return 2
    elif (distance0_1 ** 2 + distance1_4 ** 2 -distance0_4 ** 2)/(2 * distance0_1 * distance1_4)>=-0.7 and (distance0_5/distance0_8)<0.8 and (distance0_9/distance0_12)<0.8 and (distance0_13/distance0_16)>=0.8 and (distance0_17/distance0_20)>=0.8:
        return 3
    else:
        return 0


# 定义一个函数来比较两个手势
def compare_gestures(user, sys):
    if user:
        if user==sys:
            return "Draw"
        elif user-sys==1 or user-sys==-2:
            return "You win!"
        else:
            return "You Lose!"
    else:
        return ""

# 初始化MediaPipe Hands模型
mpHands = solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=1,
                      model_complexity=1,
                      min_detection_confidence=0.3,
                      min_tracking_confidence=0.3)

#手势列表
gestures=['None','Rock','Paper','Scissors']

# 初始化摄像头
cap = cv2.VideoCapture(0)

#sys代表系统手势，flag记录用户手势变化
sys,flag=0,0

while True:
    #user为用户手势
    user = 0

    # 读取摄像头帧
    ret, img = cap.read()

    if not ret:
        continue

    # 设置镜像
    img = cv2.flip(img,1)

    #得到对帧处理结果
    results = hands.process(img)

    # 绘制手部关键点
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            solutions.drawing_utils.draw_landmarks(img, hand_landmarks, mpHands.HAND_CONNECTIONS)
            user = gesture_recognizer(hand_landmarks)

    #当用户手势变化时，生成随机手势
    if user and flag != user:
        sys = random.randint(1, 3)
    elif user==0:
        sys=0
    flag = user


    #输出1、用户手势；2、系统手势；3、比赛结果
    cv2.putText(img, f"Your gesture: {gestures[user]}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3,
                cv2.LINE_AA)
    cv2.putText(img, f"Opponent's gesture: {gestures[sys]}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3,
                cv2.LINE_AA)
    cv2.putText(img, compare_gestures(user,sys), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3,
                cv2.LINE_AA)

    # 显示摄像头中的画面
    cv2.imshow('Rock Paper Scissors', img)

    # 按下'q'键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头和关闭窗口
cap.release()
cv2.destroyAllWindows()
