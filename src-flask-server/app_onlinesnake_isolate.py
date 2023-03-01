########################################################################################################################
# KRAFTON JUNGLE 1기 나만의 무기 만들기 프로젝트
# Project Biam.io
# by.Team dabCAT
# 박찬우 : https://github.com/pcw999
# 박현우 : https://github.com/phwGithub
# 우한봄 : https://github.com/onebom
# 이민섭 : https://github.com/InFinity-dev
########################################################################################################################
##################################### PYTHON PACKAGE IMPORT ############################################################
import math
import random
import cvzone
import cv2
import numpy as np
import mediapipe as mp
import sys
import os
from flask_restful import Resource, Api
from flask_cors import CORS
from datetime import datetime
import datetime
import time
from flask import Flask, render_template, Response, request, redirect, url_for, session
from flask_socketio import SocketIO, emit, join_room
import socket
from engineio.payload import Payload

from src.maze_manager import MazeManager

import simpleaudio as sa
import threading
import signal


# import pprint

########################################################################################################################
################################## SETTING GOLBAL VARIABLES ############################################################

Payload.max_decode_packets = 200

# PYTHON - ELECTRON VARIABLES
# This wil report the electron exe location, and not the /tmp dir where the exe
# is actually expanded and run from!
print(f"flask is running in {os.getcwd()}, __name__ is {__name__}", flush=True)
# print(f"flask/python env is {os.environ}", flush=True)
print(sys.version, flush=True)
# print(os.environ, flush=True)
# print(os.getcwd(), flush=True)
# print("User's Environment variable:")
# pprint.pprint(dict(os.environ), width = 1)

base_dir = '.'
if hasattr(sys, '_MEIPASS'):
    print('detected bundled mode', sys._MEIPASS)
    base_dir = os.path.join(sys._MEIPASS)

app = Flask(__name__, static_folder=os.path.join(base_dir, 'static'),
            template_folder=os.path.join(base_dir, 'templates'))

app.config['SECRET_KEY'] = "roomfitisdead"
app.config['DEBUG'] = True  # true will cause double load on startup
app.config['EXPLAIN_TEMPLATE_LOADING'] = False  # won't work unless debug is on

socketio = SocketIO(app, cors_allowed_origins='*')

CORS(app, origins='http://localhost:5000')

api = Api(app)

# Setting Path to food.png
pathFood = './src-flask-server/static/food.png'

opponent_data = {}  # 상대 데이터 (현재 손위치, 현재 뱀위치)
gameover_flag = False  # ^^ 게임오버
now_my_room = ""  # 현재 내가 있는 방
now_my_sid = ""  # 현재 나의 sid
MY_PORT = 0  # socket_bind를 위한 내 포트 번호
user_number = 0 # 1p, 2p를 나타내는 번호
user_move = False
game_over_for_debug = False

############################################################ 아마도 자바스크립트로 HTML단에서 처리 예정
# 배경음악이나 버튼음은 자바스크립트, 게임오버나 스킬 사용 효과음은 파이썬
# Global Flag for BGM status
bgm_play_obj = None
# SETTING BGM PATH
bgm_path = './src-flask-server/static/bgm/main.wav'
vfx_1_path = './src-flask-server/static/bgm/curSelect.wav'
vfx_2_path = './src-flask-server/static/bgm/eatFood.wav'
vfx_3_path = './src-flask-server/static/bgm/boost.wav'
vfx_4_path = './src-flask-server/static/bgm/gameOver.wav'
vfx_5_path = './src-flask-server/static/bgm/stageWin.wav'
def play_bgm():
    global bgm_play_obj
    bgm_wave_obj = sa.WaveObject.from_wave_file(bgm_path)
    bgm_play_obj = bgm_wave_obj.play()
    bgm_play_obj.wait_done()
def stop_music_exit(signal, frame):
    global bgm_play_obj
    if bgm_play_obj is not None:
        bgm_play_obj.stop()
    exit(0)
def stop_bgm():
    global bgm_play_obj
    if bgm_play_obj is not None:
        bgm_play_obj.stop()
# Create a new thread for each sound effect selected by the user
def play_selected_sfx(track):
    sfx_wave_obj = sa.WaveObject.from_wave_file(track)
    sfx_play_obj = sfx_wave_obj.play()
    sfx_play_obj.wait_done()
# Create a thread for the BGM
bgm_thread = threading.Thread(target=play_bgm)
# Register the signal handler for SIGINT (Ctrl-C)
signal.signal(signal.SIGINT, stop_music_exit)
############################################################

########################################################################################################################
################################ Mediapipe Detecting Module ############################################################
class HandDetector:
    """
    Finds Hands using the mediapipe library. Exports the landmarks
    in pixel format. Adds extra functionalities like finding how
    many fingers are up or the distance between two fingers. Also
    provides bounding box info of the hand found.
    """

    def __init__(self, mode=False, maxHands=1, detectionCon=0.8, minTrackCon=0.5):
        """
        :param mode: In static mode, detection is done on each image: slower
        :param maxHands: Maximum number of hands to detect
        :param detectionCon: Minimum Detection Confidence Threshold
        :param minTrackCon: Minimum Tracking Confidence Threshold
        """
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.minTrackCon = minTrackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.minTrackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.fingers = []
        self.lmList = []

    def findHands(self, img, draw=True, flipType=True):
        """
        Finds hands in a BGR image.
        :param img: Image to find the hands in.
        :param draw: Flag to draw the output on the image.
        :return: Image with or without drawings
        """

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        allHands = []
        h, w, c = img.shape
        if self.results.multi_hand_landmarks:
            for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                myHand = {}
                ## lmList
                mylmList = []
                xList = []
                yList = []
                for id, lm in enumerate(handLms.landmark):
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                    mylmList.append([px, py, pz])
                    xList.append(px)
                    yList.append(py)

                ## bbox
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                boxW, boxH = xmax - xmin, ymax - ymin
                bbox = xmin, ymin, boxW, boxH
                cx, cy = bbox[0] + (bbox[2] // 2), \
                         bbox[1] + (bbox[3] // 2)

                myHand["lmList"] = mylmList
                myHand["bbox"] = bbox
                myHand["center"] = (cx, cy)

                if flipType:
                    if handType.classification[0].label == "Right":
                        myHand["type"] = "Left"
                    else:
                        myHand["type"] = "Right"
                else:
                    myHand["type"] = handType.classification[0].label
                allHands.append(myHand)

        return allHands

    def drawHands(self, img):
        img2=img.copy()
        if self.results.multi_hand_landmarks:
            for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                self.mpDraw.draw_landmarks(img2, handLms, self.mpHands.HAND_CONNECTIONS)
        return img2

    def fingersUp(self, myHand):
        """
        Finds how many fingers are open and returns in a list.
        Considers left and right hands separately
        :return: List of which fingers are up
        """
        myHandType = myHand["type"]
        myLmList = myHand["lmList"]
        if self.results.multi_hand_landmarks:
            fingers = []
            # Thumb
            if myHandType == "Right":
                if myLmList[self.tipIds[0]][0] > myLmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                if myLmList[self.tipIds[0]][0] < myLmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # 4 Fingers
            for id in range(1, 5):
                if myLmList[self.tipIds[id]][1] < myLmList[self.tipIds[id] - 2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        return fingers

    def findDistance(self, p1, p2, img=None):
        """
        Find the distance between two landmarks based on their
        index numbers.
        :param p1: Point1
        :param p2: Point2
        :param img: Image to draw on.
        :param draw: Flag to draw the output on the image.
        :return: Distance between the points
                 Image with output drawn
                 Line information
        """

        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        info = (x1, y1, x2, y2, cx, cy)
        if img is not None:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            return length, info, img
        else:
            return length, info


########################################################################################################################
################################## SNAKE GAME LOGIC SECTION ############################################################
# video setting
cap = cv2.VideoCapture(0)

# Ubuntu YUYV cam setting low frame rate problem fixed
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

cap.set(3, 1280)
cap.set(4, 720)
cap.set(cv2.CAP_PROP_FPS, 30) # TODO : 영향 확인하기, 시간 탐지 기법 중 하나가 프레임이라 프레임 맞춰줌
fps = cap.get(cv2.CAP_PROP_FPS)

# Color templates
red = (0, 0, 255)  # red
megenta = (255, 0, 255)  # magenta
green = (0, 255, 0)  # green
yellow = (0, 255, 255)  # yellow
cyan = (255, 255, 0)  # cyan
detector = HandDetector(detectionCon=0.5, maxHands=1)


class SnakeGameClass:
    # 생성자, class를 선언하면서 기본 변수들을 설정함
    def __init__(self, pathFood):
        self.points = []  # all points of the snake
        self.lengths = []  # distance between each point
        self.currentLength = 0  # total length of the snake
        self.allowedLength = 150  # total allowed Length
        self.previousHead = (int), (int) # TODO 이거 됨 ?

        self.speed = 5
        self.minspeed=10
        self.maxspeed=math.hypot(1280, 720) / 10
        self.velocityX = random.choice([-1, 0, 1])
        self.velocityY = random.choice([-1, 1])

        self.imgFood = cv2.imread(pathFood, cv2.IMREAD_UNCHANGED)
        self.hFood, self.wFood, _ = self.imgFood.shape
        self.foodPoint = 640, 360

        self.score = 0
        self.opp_score = 0
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.opp_addr = ()
        self.is_udp = False
        self.udp_count = 0
        self.foodOnOff = True
        self.multi = True

        self.maze_start = [[],[]]
        self.maze_end = [[],[]]
        self.maze_map = np.array([])
        self.passStart=False
        self.passMid=False

        self.maze_img=np.array([0])

    def global_intialize(self):
        global user_number
        self.points = []  # all points of the snake
        self.lengths = []  # distance between each point
        self.currentLength = 0  # total length of the snake
        self.allowedLength = 150  # total allowed Length
        self.previousHead = (int), (int) # TODO 이거 됨 ?

        self.speed = 5
        self.minspeed=10
        self.maxspeed=math.hypot(1280, 720) / 10
        self.velocityX = random.choice([-1, 0, 1])
        self.velocityY = random.choice([-1, 1])

        self.hFood, self.wFood, _ = self.imgFood.shape
        self.foodPoint = 640, 360

        self.score = 0
        self.opp_score = 0
        self.opp_addr = ()
        self.is_udp = False
        self.udp_count = 0
        self.foodOnOff = True
        self.multi = True

        self.maze_start = [[],[]]
        self.maze_end = [[],[]]
        self.maze_map = np.array([])
        self.passStart=False
        user_number = 0

    def ccw(self, p, a, b):
        s = p[0] * a[1] + a[0] * b[1] + b[0] * p[1]
        s -= (p[1] * a[0] + a[1] * b[0] + b[1] * p[0])

        if s > 0 :
            return 1
        elif s == 0 :
            return 0
        else :
            return -1

    def segmentIntersects(self, p1_a, p1_b, p2_a, p2_b):
        ab = self.ccw(p1_a, p1_b, p2_a) * self.ccw(p1_a, p1_b, p2_b)
        cd = self.ccw(p2_a, p2_b, p1_a) * self.ccw(p2_a, p2_b, p1_b)

        if (ab == 0 and cd == 0):
            if (p1_a[0] > p1_b[0] or p1_a[1] > p1_b[1]):
                p1_a, p1_b = p1_b, p1_a
            if (p2_a[0] > p2_b[0] or p2_a[1] > p2_b[1]):
                p2_a, p2_b = p2_b, p2_a
            return (p2_a[0] <= p1_b[0] and p2_a[1] <= p1_b[1]) and (p1_a[0] <= p2_b[0] and p1_a[1] <= p2_b[1])

        return ab <= 0 and cd <= 0

    def isCollision(self, u1_head_pt, u2_pts):
        if not u2_pts:
            return False
        p1_a, p1_b = u1_head_pt[0], u1_head_pt[1]

        for u2_pt in u2_pts:
            p2_a, p2_b = u2_pt[0], u2_pt[1]
            if self.segmentIntersects(p1_a, p1_b, p2_a, p2_b):
                print(p1_a, p1_b, p2_a, p2_b)
                return True
        return False

    def maze_collision(self , head_pt, previous_pt):
        head_pt=np.array(head_pt).astype(int)
        # if self.maze_map[int(head_pt[1]),int(head_pt[0])]==1:
        #   return True
        pt_a = np.array(previous_pt).astype(int)
        line_norm=np.linalg.norm(pt_a - head_pt).astype(int)
        points_on_line = np.linspace(pt_a, head_pt, line_norm)
        for p in points_on_line:
            # print(p)
            if self.maze_map[int(p[1]),int(p[0])]==1:
                return True
        return False

    # maze 초기화
    def maze_initialize(self):
        self.maze_start, self.maze_mid, self.maze_end,self.maze_map = create_maze(720-300, 1280-300, 5, 12)
        self.maze_map=np.pad(self.maze_map, ((150,150),(150,150)), 'constant', constant_values=0)
        self.maze_img=self.create_maze_image()

        self.previousHead = (0, 360)
        self.velocityX = 0
        self.velocityY = 0
        self.points = []
        self.maxspeed=30
        self.passStart=False
        self.passMid=False

    def menu_initialize(self):
        self.previousHead = (0, 360)
        self.velocityX = 0
        self.velocityY = 0
        self.points = []

    def testbed_initialize(self):
        self.previousHead = (0, 360)
        self.velocityX = 0
        self.velocityY = 0
        self.points = []
        self.foodOnOff = True
        self.multi = False

    def draw_snakes(self, imgMain, points, score, isMe):

        bodercolor = cyan
        maincolor = red

        if isMe:
            bodercolor = megenta
            maincolor = green
            # Draw Score
            # cvzone.putTextRect(imgMain, f'Score: {score}', [0, 40],
            #                    scale=3, thickness=3, offset=10)

        # Change hue every 100ms
        change_interval = 100

        hue = int(time.time() * change_interval % 180) # TODO : 마지막에 성능 부족 시 아낄 수 있음
        rainbow = np.array([hue, 255, 255], dtype=np.uint8)
        rainbow = cv2.cvtColor(np.array([[rainbow]]), cv2.COLOR_HSV2BGR)[0, 0]
        # Convert headcolor to tuple of integers
        rainbow = tuple(map(int, rainbow))

        # Draw Snake
        # TODO : 아이템 먹으면 무지개 색으로 변하게?
        pts = np.array(points, np.int32)
        if len(pts.shape) == 3:
            pts = pts[:, 1]
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(imgMain, np.int32([pts]), False, maincolor, 15)

        if points:
            cv2.circle(imgMain, points[-1][1], 20, bodercolor, cv2.FILLED)
            cv2.circle(imgMain, points[-1][1], 15, rainbow, cv2.FILLED)

        return imgMain

    def draw_Food(self, imgMain):
        rx, ry = self.foodPoint
        socketio.emit('foodPoint', {'food_x': rx, 'food_y': ry})
        imgMain = cvzone.overlayPNG(imgMain, self.imgFood, (rx - self.wFood // 2, ry - self.hFood // 2))

        return imgMain

    ############################################################
    def create_maze_image(self):
        img=np.zeros((720,1280,3), dtype=np.uint8)

        img[np.where(self.maze_map == 1)] = (0, 0, 255)
        img[np.where(self.maze_map == 2)] = (0, 255, 255)
        img[np.where(self.maze_map == 3)] = (255,0, 255)
        return img

    # 내 뱀 상황 업데이트 - maze play에서
    def my_snake_update_mazeVer(self, HandPoints):
        px, py = self.previousHead
        s_speed = 30
        cx, cy = self.set_snake_speed(HandPoints, s_speed)

        self.points.append([[px, py], [cx, cy]])
        distance = math.hypot(cx - px, cy - py)
        self.lengths.append(distance)
        self.currentLength += distance
        self.previousHead = cx, cy

        self.length_reduction()
        if self.maze_collision([cx,cy],[px, py]):
            self.passStart=False
            self.passMid=False
            self.execute()

        # start point 시작!
        start_pt1, start_pt2 = self.maze_start
        if (start_pt1[0] <= cx <= start_pt2[0]) and (start_pt1[1] <= cy <= start_pt2[1]):
            self.passStart=True

        # 중간 point 패스!
        mid_pt1, mid_pt2 = self.maze_mid
        if (mid_pt1[0] <= cx <= mid_pt2[0]) and (mid_pt1[1] <= cy <= mid_pt2[1]):
            if self.passStart:
                self.passMid=True

        # end point 도달
        end_pt1, end_pt2 = self.maze_end
        # print(f"end point : 1-{end_pt1}, 2-{end_pt2}")
        if (end_pt1[0] <= cx <= end_pt2[0]) and (end_pt1[1] <= cy <= end_pt2[1]):
            if self.passStart and self.passMid:
                self.maze_initialize()
                # 시간 제한 넣는다면 그것도 다시 돌리기
                time.sleep(3)



    # 내 뱀 상황 업데이트
    def my_snake_update(self, HandPoints, opp_bodys):
        px, py = self.previousHead

        s_speed = 30
        cx, cy = self.set_snake_speed(HandPoints, s_speed)
        socketio.emit('finger_cordinate', {'head_x': cx, 'head_y': cy})

        self.points.append([[px, py], [cx, cy]])

        distance = math.hypot(cx - px, cy - py)
        self.lengths.append(distance)
        self.currentLength += distance
        self.previousHead = cx, cy

        self.length_reduction()

        if self.foodOnOff:
            self.check_snake_eating(cx, cy)

        self.send_data_to_opp()
        self.send_data_to_html()

        if self.is_udp:
            self.receive_data_from_opp()

        if len(self.points) != 0: #out of range 용 성능 애바면 좀;;
            if self.isCollision(self.points[-1], opp_bodys):
                global user_move
                if user_move:
                    self.execute()
        else:
            print('point가 텅텅 !')


    ################################## VECTORING SPEED METHOD ##########################################################
    # def set_snake_speed(self, HandPoints, s_speed):
    #   px, py = self.previousHead
    #   # ----HandsPoint moving ----
    #   if HandPoints:
    #       m_x, m_y = HandPoints
    #       dx = m_x - px  # -1~1
    #       dy = m_y - py
    #
    #
    #       # head로부터 handpoint가 근접하면 이전 direction을 따름
    #       if math.hypot(dx, dy) < 5:
    #           self.speed=5 # 최소속도
    #       else:
    #           if math.hypot(dx, dy) > 50:
    #               self.speed=50 #최대속도
    #           else:
    #               self.speed = math.hypot(dx, dy)
    #
    #       # 벡터 합 생성,크기가 1인 방향 벡터
    #       if dx!=0 and dy!=0:
    #         self.velocityX = dx/math.sqrt(dx**2+dy**2)
    #         self.velocityY = dy/math.sqrt(dx**2+dy**2)
    #
    #   else:
    #       self.speed=5
    #
    #   cx = round(px + self.velocityX*self.speed)
    #   cy = round(py + self.velocityY*self.speed)
    #   # ----HandsPoint moving ----end
    #   if cx < 0 or cx > 1280 or cy < 0 or cy > 720:
    #     if cx < 0: cx = 0
    #     if cx > 1280: cx = 1280
    #     if cy < 0: cy = 0
    #     if cy > 720: cy = 720
    #
    #   if cx == 0 or cx == 1280:
    #     self.velocityX = -self.velocityX
    #   if cy == 0 or cy == 720:
    #     self.velocityY = -self.velocityY
    #
    #   return cx, cy
    ####################################################################################################################

    def set_snake_speed(self, HandPoints, s_speed):
        px, py = self.previousHead
        # ----HandsPoint moving ----
        s_speed = 20
        if HandPoints:
            m_x, m_y = HandPoints
            dx = m_x - px  # -1~1
            dy = m_y - py

            # speed 범위: 0~1460
            if math.hypot(dx, dy) > self.maxspeed : # 146
                self.speed = self.maxspeed
            elif math.hypot(dx, dy) < self.minspeed:
                self.speed = self.minspeed
            else:
                self.speed = math.hypot(dx, dy)

            if dx != 0:
                self.velocityX = dx / 1280
            if dy != 0:
                self.velocityY = dy / 720

            # print(self.velocityX)
            # print(self.velocityY)

        else:
            self.speed = self.minspeed

        cx = round(px + self.velocityX * self.speed)
        cy = round(py + self.velocityY * self.speed)
        # ----HandsPoint moving ----end
        if cx < 0 or cx > 1280 or cy < 0 or cy > 720:
            if cx < 0: cx = 0
            if cx > 1280: cx = 1280
            if cy < 0: cy = 0
            if cy > 720: cy = 720

        if cx == 0 or cx == 1280:
            self.velocityX = -self.velocityX
        if cy == 0 or cy == 720:
            self.velocityY = -self.velocityY

        return cx, cy

    # 뱀 길이 조정
    def length_reduction(self):
        if self.currentLength > self.allowedLength:
            for i, length in enumerate(self.lengths):
                self.currentLength -= length
                self.lengths = self.lengths[1:]
                self.points = self.points[1:]

                if self.currentLength < self.allowedLength:
                    break

    # 뱀 식사 여부 확인
    def check_snake_eating(self, cx, cy):
        rx, ry = self.foodPoint
        if (rx - (self.wFood // 2) < cx < rx + (self.wFood // 2)) and (
                ry - (self.hFood // 2) < cy < ry + (self.hFood // 2)):
            sfx_thread = threading.Thread(target=play_selected_sfx, args=(vfx_2_path,))
            sfx_thread.start()
            self.allowedLength += 50
            self.score += 1

            if self.multi:
                self.foodOnOff = False
                socketio.emit('user_ate_food', {'score': self.score})
            else:
                self.foodPoint = random.randint(100, 1000), random.randint(100, 600)

    # 뱀이 충돌했을때
    def execute(self):
        global user_move
        global user_number
        global game_over_for_debug
        self.points = []  # all points of the snake
        self.lengths = []  # distance between each point
        self.currentLength = 0  # total length of the snake
        self.allowedLength = 150  # total allowed Length
        if user_number == 1:
            self.previousHead = 100, 180
        elif user_number == 2:
            self.previousHead = 1180, 540
        else:
            self.previousHead = 0, 360
        user_move = False
        game_over_for_debug = True
        socketio.emit('gameover')

    def update_mazeVer(self, imgMain, HandPoints):
        global gameover_flag

        if self.gameOver:
            gameover_flag = False
        else:
            self.my_snake_update_mazeVer(HandPoints)
            imgMain = self.draw_snakes(imgMain, self.points, self.score, 1)

        return imgMain

    # 송출될 프레임 업데이트
    def update(self, imgMain, HandPoints):
        global gameover_flag, opponent_data

        opp_bodys = []
        # 0 이면 상대 뱀
        if opponent_data:
            opp_bodys = opponent_data['opp_body_node']
        imgMain = self.draw_snakes(imgMain, opp_bodys, self.opp_score, 0)

        # update and draw own snake
        self.my_snake_update(HandPoints, opp_bodys)
        imgMain = self.draw_Food(imgMain)
        # 1 이면 내 뱀
        imgMain = self.draw_snakes(imgMain, self.points, self.score, 1)

        return imgMain

    # Menu 화면에서 쓰일 검은 배경 뱀
    def update_blackbg(self, imgMain, HandPoints):
        global gameover_flag, opponent_data

        # update and draw own snake
        self.my_snake_update(HandPoints, [])
        imgMain = self.draw_snakes(imgMain, self.points, self.score, 1)

        return imgMain

    # 통신 관련 변수 설정
    def set_socket(self, my_port, opp_ip, opp_port):
        self.sock.bind(('0.0.0.0', int(my_port)))
        self.sock.settimeout(0.01) # TODO 만약 udp, 서버 선택 오류 시 다시 0.02로
        self.opp_addr = (opp_ip, int(opp_port))

    # 데이터 전송
    def send_data_to_opp(self):
        if self.is_udp:
            data_set = str(self.points)
            self.sock.sendto(data_set.encode(), self.opp_addr)
        else:
            socketio.emit('game_data', {'body_node': self.points})

    def send_data_to_html(self):
        socketio.emit('game_data_for_debug', {'score': self.score, 'fps': fps})

    # 데이터 수신 (udp 통신 일때만 사용)
    def receive_data_from_opp(self):
        global opponent_data

        try:
            data, _ = self.sock.recvfrom(15000)
            decode_data = data.decode()
            if decode_data[0] == '[':
                opponent_data['opp_body_node'] = eval(decode_data)
                self.udp_count = 0
            else:
                test_code = decode_data
                self.sock.sendto(test_code.encode(), self.opp_addr)
        except socket.timeout:
            self.udp_count += 1
            if self.udp_count > 25:
                socketio.emit('opponent_escaped')

    # udp로 통신할지 말지
    def test_connect(self, sid):
        a = 0
        b = 0
        test_code = str(sid)

        for i in range(50):
            if i % 2 == 0:
                test_code = str(sid)
            self.sock.sendto(test_code.encode(), self.opp_addr)
            try:
                data, _ = self.sock.recvfrom(600)
                test_code = data.decode()
                if test_code == str(sid):
                    b += 1
            except socket.timeout:
                a += 1

        if a != 50 and b != 0:
            self.is_udp = True

        print(f"connection MODE : {self.is_udp} / a = {a}, b = {b}")
        socketio.emit('NetworkMode', {'UDP': self.is_udp})

    # 소멸자 소켓 bind 해제
    def __del__(self):
        global opponent_data
        opponent_data = {}
        self.sock.close()

class MultiSnake:
    def __init__(self, pathFood):
        self.points = []
        self.lengths = []
        self.currentLength = 0
        self.allowedLength = 150
        self.previousHead = 0, 0
        self.score = 0
        self.opp_score = 0

        self.speed = 5
        self.minspeed=10
        self.maxspeed=math.hypot(1280, 720) / 10
        self.velocityX = random.choice([-1, 0, 1])
        self.velocityY = random.choice([-1, 1])

        self.imgFood = cv2.imread(pathFood, cv2.IMREAD_UNCHANGED)
        self.hFood, self.wFood, _ = self.imgFood.shape
        self.foodPoint = 640, 360
        self.foodOnOff = True

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.opp_addr = ()
        self.is_udp = False
        self.udp_count = 0

        self.gameover = False

    def ccw(self, p, a, b):
        s = p[0] * a[1] + a[0] * b[1] + b[0] * p[1]
        s -= (p[1] * a[0] + a[1] * b[0] + b[1] * p[0])

        if s > 0 :
            return 1
        elif s == 0 :
            return 0
        else :
            return -1

    def segmentIntersects(self, p1_a, p1_b, p2_a, p2_b):
        ab = self.ccw(p1_a, p1_b, p2_a) * self.ccw(p1_a, p1_b, p2_b)
        cd = self.ccw(p2_a, p2_b, p1_a) * self.ccw(p2_a, p2_b, p1_b)

        if (ab == 0 and cd == 0):
            if (p1_a[0] > p1_b[0] or p1_a[1] > p1_b[1]):
                p1_a, p1_b = p1_b, p1_a
            if (p2_a[0] > p2_b[0] or p2_a[1] > p2_b[1]):
                p2_a, p2_b = p2_b, p2_a
            return (p2_a[0] <= p1_b[0] and p2_a[1] <= p1_b[1]) and (p1_a[0] <= p2_b[0] and p1_a[1] <= p2_b[1])

    def isCollision(self, u1_head_pt, u2_pts):
        if not u2_pts:
            return False
        p1_a, p1_b = u1_head_pt[0], u1_head_pt[1]

        for u2_pt in u2_pts:
            p2_a, p2_b = u2_pt[0], u2_pt[1]
            if self.segmentIntersects(p1_a, p1_b, p2_a, p2_b):
                print(p1_a, p1_b, p2_a, p2_b)
                return True
        return False

    def draw_snakes(self, imgMain, points, isMe):
        bodercolor = cyan
        maincolor = red

        if isMe:
            bodercolor = megenta
            maincolor = green

        # Change hue every 100ms
        change_interval = 100

        hue = int(time.time() * change_interval % 180) # TODO : 마지막에 성능 부족 시 아낄 수 있음
        rainbow = np.array([hue, 255, 255], dtype=np.uint8)
        rainbow = cv2.cvtColor(np.array([[rainbow]]), cv2.COLOR_HSV2BGR)[0, 0]
        # Convert headcolor to tuple of integers
        rainbow = tuple(map(int, rainbow))

        # Draw Snake
        # TODO : 아이템 먹으면 무지개 색으로 변하게?
        pts = np.array(points, np.int32)
        if len(pts.shape) == 3:
            pts = pts[:, 1]
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(imgMain, np.int32([pts]), False, maincolor, 15)

        if points:
            cv2.circle(imgMain, points[-1][1], 20, bodercolor, cv2.FILLED)
            cv2.circle(imgMain, points[-1][1], 15, rainbow, cv2.FILLED)

        return imgMain

    def draw_Food(self, imgMain):
        rx, ry = self.foodPoint
        socketio.emit('foodPoint', {'food_x': rx, 'food_y': ry})
        imgMain = cvzone.overlayPNG(imgMain, self.imgFood, (rx - self.wFood // 2, ry - self.hFood // 2))

        return imgMain

    def my_snake_update(self, HandPoints, opp_bodys):
        px, py = self.previousHead

        s_speed = 30
        cx, cy = self.set_snake_speed(HandPoints, s_speed)
        socketio.emit('finger_cordinate', {'head_x': cx, 'head_y': cy})

        self.points.append([[px, py], [cx, cy]])

        distance = math.hypot(cx - px, cy - py)
        self.lengths.append(distance)
        self.currentLength += distance
        self.previousHead = cx, cy

        self.length_reduction()

        if self.foodOnOff:
            self.check_snake_eating(cx, cy)

        self.send_data_to_opp()
        self.send_data_to_html()

        if self.is_udp:
            self.receive_data_from_opp()

        if len(self.points) != 0: #out of range 용 성능 애바면 좀;;
            if self.isCollision(self.points[-1], opp_bodys):
                self.execute()
        else:
            print('point가 텅텅 !')

    def set_snake_speed(self, HandPoints, s_speed):
        px, py = self.previousHead
        # ----HandsPoint moving ----
        s_speed = 20
        if HandPoints:
            m_x, m_y = HandPoints
            dx = m_x - px  # -1~1
            dy = m_y - py

            # speed 범위: 0~1460
            if math.hypot(dx, dy) > self.maxspeed : # 146
                self.speed = self.maxspeed
            elif math.hypot(dx, dy) < self.minspeed:
                self.speed = self.minspeed
            else:
                self.speed = math.hypot(dx, dy)

            if dx != 0:
                self.velocityX = dx / 1280
            if dy != 0:
                self.velocityY = dy / 720

        else:
            self.speed = self.minspeed

        cx = round(px + self.velocityX * self.speed)
        cy = round(py + self.velocityY * self.speed)
        # ----HandsPoint moving ----end
        if cx < 0 or cx > 1280 or cy < 0 or cy > 720:
            if cx < 0: cx = 0
            if cx > 1280: cx = 1280
            if cy < 0: cy = 0
            if cy > 720: cy = 720

        if cx == 0 or cx == 1280:
            self.velocityX = -self.velocityX
        if cy == 0 or cy == 720:
            self.velocityY = -self.velocityY

        return cx, cy

    def length_reduction(self):
        if self.currentLength > self.allowedLength:
            for i, length in enumerate(self.lengths):
                self.currentLength -= length
                self.lengths = self.lengths[1:]
                self.points = self.points[1:]

                if self.currentLength < self.allowedLength:
                    break

    def check_snake_eating(self, cx, cy):
        rx, ry = self.foodPoint
        if (rx - (self.wFood // 2) < cx < rx + (self.wFood // 2)) and (
                ry - (self.hFood // 2) < cy < ry + (self.hFood // 2)):
            sfx_thread = threading.Thread(target=play_selected_sfx, args=(vfx_2_path,))
            sfx_thread.start()
            self.allowedLength += 50
            self.score += 1

            self.foodOnOff = False
            socketio.emit('user_ate_food', {'score': self.score})

    def execute(self):
        self.points = []  # all points of the snake
        self.lengths = []  # distance between each point
        self.currentLength = 0  # total length of the snake
        self.allowedLength = 150  # total allowed Length

        socketio.emit('gameover')
        self.gameover = True
        socketio.emit('gameover')

    def update(self, imgMain, HandPoints):
        global opponent_data

        opp_bodys = []
        # 0 이면 상대 뱀
        if opponent_data:
            opp_bodys = opponent_data['opp_body_node']
        imgMain = self.draw_snakes(imgMain, opp_bodys, 0)

        # update and draw own snake
        self.my_snake_update(HandPoints, opp_bodys)
        imgMain = self.draw_Food(imgMain)
        # 1 이면 내 뱀
        imgMain = self.draw_snakes(imgMain, self.points, 1)

        return imgMain

    def set_socket(self, my_port, opp_ip, opp_port):
        self.sock.bind(('0.0.0.0', int(my_port)))
        self.sock.settimeout(0.02) # TODO 만약 udp, 서버 선택 오류 시 다시 0.02로
        self.opp_addr = (opp_ip, int(opp_port))

    def send_data_to_opp(self):
        if self.is_udp:
            data_set = str(self.points)
            self.sock.sendto(data_set.encode(), self.opp_addr)
        else:
            socketio.emit('game_data', {'body_node': self.points})

    def send_data_to_html(self):
        socketio.emit('game_data_for_debug', {'score': self.score, 'fps': fps})

    def receive_data_from_opp(self):
        global opponent_data

        try:
            data, _ = self.sock.recvfrom(15000)
            decode_data = data.decode()
            if decode_data[0] == '[':
                opponent_data['opp_body_node'] = eval(decode_data)
                self.udp_count = 0
            else:
                test_code = decode_data
                self.sock.sendto(test_code.encode(), self.opp_addr)
        except socket.timeout:
            self.udp_count += 1
            if self.udp_count > 25:
                socketio.emit('opponent_escaped')

    def test_connect(self, sid):
        a = 0
        b = 0
        test_code = str(sid)

        for i in range(60):
            if i % 2 == 0:
                test_code = str(sid)
            self.sock.sendto(test_code.encode(), self.opp_addr)
            try:
                data, _ = self.sock.recvfrom(600)
                test_code = data.decode()
                if test_code == str(sid):
                    b += 1
            except socket.timeout:
                a += 1

        if a != 60 and b != 0:
            self.is_udp = True

        print(f"connection MODE : {self.is_udp} / a = {a}, b = {b}")
        socketio.emit('NetworkMode', {'UDP': self.is_udp})

    def __del__(self):
        global opponent_data
        opponent_data = {}
        self.sock.close()



########################################################################################################################
######################################## FLASK APP ROUTINGS ############################################################

game = SnakeGameClass(pathFood)
multigame = MultiSnake(pathFood)

# Defualt Root Routing for Flask Server Check
@api.resource('/')
class HelloWorld(Resource):
    def get(self):
        print(f'Electron GET Requested from HTML', flush=True)
        data = {'Flask 서버 클라이언트 to Electron': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        return data


# Game Main Menu
@app.route("/index", methods=["GET", "POST"])
def index():
    return render_template("index.html")


@app.route('/testbed')
def testbed():
    return render_template("testbed.html")

@app.route('/mazerunner')
def mazerunner():
    return render_template("mazerunner.html")


# Game Screen
@app.route("/enter_snake", methods=["GET", "POST"])
def enter_snake():
    global now_my_sid
    global now_my_room
    global multigame

    now_my_room = request.args.get('room_id')
    now_my_sid = request.args.get('sid')
    print(now_my_room, now_my_sid)

    multigame = MultiSnake(pathFood)

    return render_template("snake.html", room_id=now_my_room, sid=now_my_sid)


########################################################################################################################
############## SERVER SOCKET AND PEER TO PEER ESTABLISHMENT ############################################################

# 페이지에서 로컬 flask 서버와 소켓 통신 개시 되었을때 자동으로 실행
@socketio.on('connect')
def test_connect():
    print('Client connected!!!')


# 페이지에서 로컬 flask 서버와 소켓 통신 종료 되었을때 자동으로 실행
@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected!!!')


# 현재 내 포트 번호 요청
@socketio.on('my_port')
def my_port(data):
    global MY_PORT

    MY_PORT = data['my_port']


# webpage로 부터 받은 상대방 주소 (socket 통신에 사용)
@socketio.on('opponent_address')
def set_address(data):
    global MY_PORT
    global multigame
    global user_number
    opp_ip = data['ip_addr']
    opp_port = data['port']
    sid = data['sid']
    user_number = data['user_number']

    multigame.set_socket(MY_PORT, opp_ip, opp_port)
    multigame.test_connect(sid)


# socketio로 받은 상대방 정보
@socketio.on('opp_data_transfer')
def opp_data_transfer(data):
    global opponent_data
    opponent_data = data['data']


# socketio로 받은 먹이 위치
@socketio.on('set_food_location')
def set_food_loc(data):
    global multigame
    multigame.foodPoint = data['foodPoint']
    multigame.foodOnOff = True


# socketio로 받은 먹이 위치와 상대 점수
@socketio.on('set_food_location_score')
def set_food_loc(data):
    global multigame
    multigame.foodPoint = data['foodPoint']
    multigame.opp_score = data['opp_score']
    multigame.foodOnOff = True


########################################################################################################################
######################################## MAIN GAME ROUNTING ############################################################
@app.route('/snake')
def snake():
    def generate():
        global opponent_data
        global multigame
        global user_number

        while True :
            if user_number == 1:
                multigame.previousHead = (100, 360)
                break
            elif user_number == 2:
                multigame.previousHead = (1180, 360)
                break

        while True:
            success, img = cap.read()
            img = cv2.flip(img, 1)
            hands = detector.findHands(img, flipType=False)
            img=detector.drawHands(img)

            pointIndex = []

            if hands:
                lmList = hands[0]['lmList']
                pointIndex = lmList[8][0:2]

            img = multigame.update(img, pointIndex)

            # encode the image as a JPEG string
            _, img_encoded = cv2.imencode('.jpg', img)
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + img_encoded.tobytes() + b'\r\n')

            if multigame.gameover :
                break

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


########################################################################################################################
############################### TEST BED FOR GAME LOGIC DEV ############################################################

# SETTING UP VARIABLES AND FUNCTION FOR BOT
bot_data = {'bot_head_x': 1000,
            'bot_head_y': 360,
            'bot_body_node': [],
            'currentLength': 0,
            'lengths': [],
            'bot_velocityX': random.choice([-1, 1]),
            'bot_velocityY': random.choice([-1, 1])}
bot_cnt = 0


def bot_data_update():
    global bot_data, bot_cnt

    bot_speed = 20
    px, py = bot_data['bot_head_x'], bot_data['bot_head_y']

    if px <= 0 or px >= 1280 or py <= 0 or py >= 720:
        if px < 0: px = 0
        if px > 1280: px = 1280
        if py < 0: py = 0
        if py > 720: py = 720

        if px == 0 or px == 1280:
            bot_data['bot_velocityX'] = -bot_data['bot_velocityX']
        if py == 0 or py == 720:
            bot_data['bot_velocityY'] = -bot_data['bot_velocityY']

    # 1초 마다 방향 바꾸기
    # print(bot_cnt)
    if bot_cnt == 30:
        bot_data['bot_velocityX'] = random.choice([-1, 0, 1])
        if bot_data['bot_velocityX'] == 0:
            bot_data['bot_velocityY'] = random.choice([-1, 1])
        else:
            bot_data['bot_velocityY'] = random.choice([-1, 0, 1])
        bot_cnt = 0
    bot_cnt += 1

    bot_velocityX = bot_data['bot_velocityX']
    bot_velocityY = bot_data['bot_velocityY']

    cx = round(px + bot_velocityX * bot_speed)
    cy = round(py + bot_velocityY * bot_speed)

    bot_data['bot_head_x'] = cx
    bot_data['bot_head_y'] = cy
    bot_data['bot_body_node'].append([[px, py], [cx, cy]])

    distance = math.hypot(cx - px, cy - py)
    bot_data['lengths'].append(distance)
    bot_data['currentLength'] += distance

    socketio.emit('bot_data', {'head_x': cx, 'head_y': cy})

    if bot_data['currentLength'] > 250:
        for i, length in enumerate(bot_data['lengths']):
            bot_data['currentLength'] -= length
            bot_data['lengths'] = bot_data['lengths'][1:]
            bot_data['bot_body_node'] = bot_data['bot_body_node'][1:]

            if bot_data['currentLength'] < 250:
                break

single_game = SnakeGameClass(pathFood)
# TEST BED ROUTING
@app.route('/test')
def test():
    def generate():
        global bot_data, single_game, gameover_flag
        global opponent_data
        single_game.global_intialize()
        single_game.testbed_initialize()

        ## CONFILIC FLAG HERE
        max_time_end = time.time() + 4
        cx, cy = 200, 360
        while True:
            success, img = cap.read()
            img = cv2.flip(img, 1)
            hands = detector.findHands(img, flipType=False)
            img=detector.drawHands(img)

            cx += 1
            pointIndex = [cx, cy]

            bot_data_update()
            opponent_data['opp_body_node'] = bot_data["bot_body_node"]
            # print(pointIndex)

            img = single_game.update(img, pointIndex)

            # encode the image as a JPEG string
            _, img_encoded = cv2.imencode('.jpg', img)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img_encoded.tobytes() + b'\r\n')

            if time.time() > max_time_end:
                break

        single_game.previousHead = cx, cy
        ## CONFILIC FLAG HERE

        while True:
            success, img = cap.read()
            img = cv2.flip(img, 1)
            hands = detector.findHands(img, flipType=False)
            img=detector.drawHands(img)

            pointIndex = []

            if hands:
                lmList = hands[0]['lmList']
                pointIndex = lmList[8][0:2]

            bot_data_update()
            opponent_data['opp_body_node'] = bot_data["bot_body_node"]
            # print(pointIndex)

            img = single_game.update(img, pointIndex)

            # encode the image as a JPEG string
            _, img_encoded = cv2.imencode('.jpg', img)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img_encoded.tobytes() + b'\r\n')

            if gameover_flag:
                print("game ended")
                gameover_flag = False
                time.sleep(1)
                socketio.emit('gameover', {'sid': sid})
                time.sleep(2)
                break

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Main Menu Selection
@app.route('/menu_snake')
def menu_snake():
    menu_game = SnakeGameClass(pathFood)

    menu_game.multi = False
    menu_game.foodOnOff = False
    menuimg=np.zeros((720,1280,3),dtype=np.uint8)
    menu_game.global_intialize()
    menu_game.menu_initialize()

    def generate():
        while True:
            success, img = cap.read()
            img = cv2.flip(img, 1)
            hands = detector.findHands(img, flipType=False)
            showimg=detector.drawHands(menuimg)
            pointIndex = []

            if hands:
                lmList = hands[0]['lmList']
                pointIndex = lmList[8][0:2]

            showimg = menu_game.update_blackbg(showimg, pointIndex)
            # encode the image as a JPEG string
            _, img_encoded = cv2.imencode('.jpg', showimg)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img_encoded.tobytes() + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


def create_maze(image_h, image_w, block_rows, block_cols):
    manager = MazeManager()
    maze = manager.add_maze(block_rows, block_cols)
    manager.solve_maze(maze.id, "DepthFirstBacktracker")

    wall_map = np.zeros((image_h, image_w))  # (h,w)
    block_h = image_h // block_rows
    block_w = image_w // block_cols

    start = [[],[]]
    end = [[], []]
    r = 2

    for i in range(block_rows):
        for j in range(block_cols):
            if maze.initial_grid[i][j].is_entry_exit == "entry":
                start = [[j * block_w + 150, i * block_h + 150], [(j+1) * block_w +150, (i+1) * block_h +150]]
                wall_map[i * block_h +10 : (i+1) * block_h - 10, j * block_w +10 : (j+1) * block_w -10] = 2
                print(f"start in create_maze: {start}")
            elif maze.initial_grid[i][j].is_entry_exit == "exit":
                end = [[j * block_w + 150, i * block_h+ 150], [(j+1) * block_w +150, (i+1) * block_h +150]]
                wall_map[i * block_h + 10 : (i+1) * block_h -10 , j * block_w + 10 : (j + 1) * block_w - 10] = 3
                print(f"end in create_maze:{end}")
            if maze.initial_grid[i][j].walls["top"]:
                if i == 0:
                    wall_map[i * block_h:i * block_h + r, j * block_w:(j + 1) * block_w] = 1
                else:
                    wall_map[i * block_h - r:i * block_h + r, j * block_w:(j + 1) * block_w] = 1
            if maze.initial_grid[i][j].walls["right"]:
                wall_map[i * block_h:(i + 1) * block_h, (j + 1) * block_w - r:(j + 1) * block_w + r] = 1
            if maze.initial_grid[i][j].walls["bottom"]:
                wall_map[(i + 1) * block_h - r:(i + 1) * block_h + r, j * block_w:(j + 1) * block_w] = 1
            if maze.initial_grid[i][j].walls["left"]:
                if j == 0:
                    wall_map[i * block_h:(i + 1) * block_h, j * block_w:j * block_w + r] = 1
                else:
                    wall_map[i * block_h:(i + 1) * block_h, j * block_w - r:j * block_w + r] = 1

    solution_nodes=maze.solution_path
    mid_goal_h=maze.solution_path[len(solution_nodes)//2][0][0]
    mid_goal_x=maze.solution_path[len(solution_nodes)//2][0][1]
    # print(len(solution_nodes))
    # print(mid_goal_h)
    # print(mid_goal_x)
    mid=[[mid_goal_x * block_w + 150, mid_goal_h * block_h + 150], [(mid_goal_x+1) * block_w +150, (mid_goal_h+1) * block_h +150]]

    return start, mid, end, wall_map


@app.route('/maze_play')
def maze_play():
    def generate():
        global gameover_flag,game

        game.multi = False
        game.maze_initialize()

        while True:
            success, img = cap.read()
            img = cv2.flip(img, 1)

            hands = detector.findHands(img, flipType=False)
            showimg=detector.drawHands(game.maze_img) # 무조건 findHands 다음

            pointIndex = []
            if hands:
                lmList = hands[0]['lmList']
                pointIndex = lmList[8][0:2]

            showimg = game.update_mazeVer(showimg, pointIndex)

            # encode the image as a JPEG string
            _, img_encoded = cv2.imencode('.jpg', showimg)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img_encoded.tobytes() + b'\r\n')

            if gameover_flag:
                print("game ended")
                gameover_flag = False
                time.sleep(1)
                socketio.emit('gameover', {'sid': sid})
                time.sleep(2)
                break

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


########################################################################################################################
########################## Legacy Electron Template Routing ############################################################
@app.route('/hello')
def hello():
    return render_template('hello.html', msg="YOU")


@app.route('/hello-vue')
def hello_vue():
    return render_template('hello-vue.html', msg="WELCOME 🌻")


########################################################################################################################
####################################### FLASK APP ARGUMENTS ############################################################

if __name__ == "__main__":
    socketio.run(app, host='localhost', port=5000, debug=False, allow_unsafe_werkzeug=True)

########################################################################################################################