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

# Network connections variables
opponent_data = {}  # 상대 데이터 (현재 손위치, 현재 뱀위치)
gameover_flag = False  # ^^ 게임오버
now_my_room = ""  # 현재 내가 있는 방
now_my_sid = ""  # 현재 나의 sid
MY_PORT = 0  # socket_bind를 위한 내 포트 번호


########################################################################################################################
################################ Mediapipe Detecting Module ############################################################
class HandDetector:
  """
  Finds Hands using the mediapipe library. Exports the landmarks
  in pixel format. Adds extra functionalities like finding how
  many fingers are up or the distance between two fingers. Also
  provides bounding box info of the hand found.
  """

  def __init__(self, mode=False, maxHands=2, detectionCon=0.5, minTrackCon=0.5):
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

        # gaussian blur value
        # [TODO] 조건문으로 가우시안 줄지말지 정하기
        sigma = 10
        img = (cv2.GaussianBlur(img, (0, 0), sigma))

        ## draw
        if draw:
          self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
          # cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
          #               (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
          #               (255, 0, 255), 2)
          cv2.putText(img, myHand["type"], (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN,
                      2, (255, 0, 255), 2)
    else:
      sigma = 10
      img = (cv2.GaussianBlur(img, (0, 0), sigma))
    if draw:
      return allHands, img
    else:
      return allHands

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
############################## SNAKE GAME LOGIC SECTION ##############################
# video setting
cap = cv2.VideoCapture(0)

# Ubuntu YUYV cam setting low frame rate problem fixed
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

cap.set(3, 1280)
cap.set(4, 720)
cap.set(cv2.CAP_PROP_FPS, 60)
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
    self.previousHead = random.randint(100, 1000), random.randint(100, 600)

    self.speed = 5
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
    self.gameOver = False

    self.multi=True


  def ccw(self, p, a, b):
    # print("확인3")
    vect_sub_ap = [a[0] - p[0], a[1] - p[1]]
    vect_sub_bp = [b[0] - p[0], b[1] - p[1]]
    return vect_sub_ap[0] * vect_sub_bp[1] - vect_sub_ap[1] * vect_sub_bp[0]

  def segmentIntersects(self, p1_a, p1_b, p2_a, p2_b):
    # print("확인2")
    ab = self.ccw(p1_a, p1_b, p2_a) * self.ccw(p1_a, p1_b, p2_b)
    cd = self.ccw(p2_a, p2_b, p1_a) * self.ccw(p2_a, p2_b, p1_b)

    if (ab == 0 and cd == 0):
      if (p1_b[0] < p1_a[0] and p1_b[1] < p1_a[1]):
        p1_a, p1_b = p1_b, p1_a
      if (p2_b[0] < p2_a[0] and p2_b[1] < p2_a[1]):
        p2_a, p2_b = p2_b, p2_a
      return not ((p1_b[0] < p2_a[0] and p1_b[1] < p2_a[1]) or (p2_b[0] < p1_a[0] and p2_b[1] < p1_a[1]))

    return ab <= 0 and cd <= 0

  def isCollision(self, u1_head_pt, u2_pts):
    # print("확인1")
    if not u2_pts:
      return False
    p1_a, p1_b = u1_head_pt[0], u1_head_pt[1]

    for u2_pt in u2_pts:
      p2_a, p2_b = u2_pt[0], u2_pt[1]
      if self.segmentIntersects(p1_a, p1_b, p2_a, p2_b):
        # print(u2_pt)
        return True
    return False

  def draw_snakes(self, imgMain, points, score, isMe):

    bodercolor = cyan
    maincolor = red

    if isMe:
      bodercolor = megenta
      maincolor = green
      # Draw Score
      cvzone.putTextRect(imgMain, f'Score: {score}', [0, 40],
                         scale=3, thickness=3, offset=10)

    # Draw Snake
    if points:
      cv2.circle(imgMain, points[-1][1], 20, bodercolor, cv2.FILLED)
      cv2.circle(imgMain, points[-1][1], 15, maincolor, cv2.FILLED)

    pts = np.array(points, np.int32)
    if len(pts.shape) == 3:
      pts = pts[:, 1]
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(imgMain, np.int32([pts]), False, maincolor, 15)

    return imgMain

  def draw_Food(self, imgMain):
    rx, ry = self.foodPoint
    imgMain = cvzone.overlayPNG(imgMain, self.imgFood, (rx - self.wFood // 2, ry - self.hFood // 2))
    return imgMain

  ############################################################
  # 내 뱀 상황 업데이트
  def my_snake_update(self, HandPoints):
    global opponent_data

    px, py = self.previousHead

    s_speed = 30
    cx, cy = self.set_snake_speed(HandPoints, s_speed)

    self.points.append([[px, py], [cx, cy]])

    distance = math.hypot(cx - px, cy - py)
    self.lengths.append(distance)
    self.currentLength += distance
    self.previousHead = cx, cy

    self.length_reduction()

    self.check_snake_eating(cx, cy)

    self.send_data_to_opp()

    if self.is_udp:
      self.receive_data_from_opp()

    # if self.isCollision(self.points[-1], o_bodys):
    #     self.execute()

  # 내 뱀이 움직이는 속도 설정
  def set_snake_speed(self, HandPoints, s_speed):
    px, py = self.previousHead
    # ----HandsPoint moving ----
    if HandPoints:
        m_x, m_y = HandPoints
        dx = m_x - px  # -1~1
        dy = m_y - py


        # head로부터 handpoint가 근접하면 이전 direction을 따름
        if math.hypot(dx, dy) < 1:
            self.speed=1 # 최소 속도
        else:
            if math.hypot(dx, dy) > 40:
                self.speed=40 #최대속도
            else:
                self.speed = math.hypot(dx, dy)

        # 벡터 합 생성,크기가 1인 방향 벡터
        if dx!=0:
          a_vx=(self.velocityX*self.speed+dx/math.sqrt(dx**2+dy**2))
          self.velocityX = dx/math.sqrt(dx**2+dy**2)
        else:
          a_vx=self.velocityX*self.speed

        if dy!=0:
          a_vy=(self.velocityY*self.speed+dy/math.sqrt(dx**2+dy**2))
          self.velocityY = dy/math.sqrt(dx**2+dy**2)
        else:
          a_vy=self.velocityY*self.speed

    else:
        a_vx=self.velocityX*self.speed
        a_vy=self.velocityY*self.speed

    cx = round(px + a_vx)
    cy = round(py + a_vy)
    # ----HandsPoint moving ----end
    if cx<0 or cx>1280 or cy< 0 or cy>720:
        if cx<0: cx=0
        if cx>1280: cx=1280
        if cy<0: cy=0
        if cy>720: cy=720

    if cx==0 or cx==1280:
        self.velocityX=-self.velocityX
    if cy== 0 or cy==720:
        self.velocityY=-self.velocityY

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
      self.allowedLength += 50
      self.score += 1

      if self.multi:
        socketio.emit('user_ate_food', {'score': self.score})
      else:
        self.foodPoint = random.randint(100, 1000), random.randint(100, 600)

  # 뱀이 충돌했을때
  def execute(self):
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Hit")
    self.gameOver = False
    self.points = []  # all points of the snake
    self.lengths = []  # distance between each point
    self.currentLength = 0  # total length of the snake
    self.allowedLength = 150  # total allowed Length
    self.previousHead = 0, 0  # previous head point

  # 송출될 프레임 업데이트
  def update(self, imgMain, HandPoints):
    global gameover_flag, opponent_data

    if self.gameOver:
      gameover_flag = False
    else:
      opp_bodys=[]
      # 0 이면 상대 뱀
      if opponent_data:
        opp_bodys=opponent_data['opp_body_node']
      imgMain = self.draw_snakes(imgMain, opp_bodys, self.opp_score, 0)

      # update and draw own snake
      self.my_snake_update(HandPoints)
      imgMain = self.draw_Food(imgMain)
      # 1 이면 내 뱀
      imgMain = self.draw_snakes(imgMain, self.points, self.score, 1)

    return imgMain

  # 통신 관련 변수 설정
  def set_socket(self, my_port, opp_ip, opp_port):
    self.sock.bind(('0.0.0.0', int(my_port)))
    self.sock.settimeout(0.02)
    self.opp_addr = (opp_ip, int(opp_port))

  # 데이터 전송
  def send_data_to_opp(self):
    if self.is_udp:
      data_set = str(self.points)
      self.sock.sendto(data_set.encode(), self.opp_addr)
    else:
      socketio.emit('game_data', {'body_node': self.points})

  # 데이터 수신 (udp 통신 일때만 사용)
  def receive_data_from_opp(self):
    global opponent_data

    try:
      data, _ = self.sock.recvfrom(15000)
      decode_data = data.decode()
      if decode_data == 'A':
        pass
      else:
        opponent_data['opp_body_node'] = eval(decode_data)
    except socket.timeout:
      pass

  # udp로 통신할지 말지
  def test_connect(self):
    a = 0

    for i in range(10):
      test_code = 'A'
      self.sock.sendto(test_code.encode(), self.opp_addr)
      try:
        data, result = self.sock.recvfrom(1000)
      except socket.timeout:
        a += 1

    if a != 0:
      self.is_udp = True
      print("UDP MODE")

  # 소멸자 소켓 bind 해제
  def __del__(self):
    global opponent_data
    opponent_data = {}
    self.sock.close()


######################################################################################
######################################## FLASK APP ROUTINGS ############################################################

game = SnakeGameClass(pathFood)

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


# Game Screen
@app.route("/enter_snake", methods=["GET", "POST"])
def enter_snake():
  global now_my_sid
  global now_my_room
  global game

  now_my_room = request.args.get('room_id')
  now_my_sid = request.args.get('sid')
  print(now_my_room, now_my_sid)

  game = SnakeGameClass(pathFood)

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
  global game
  opp_ip = data['ip_addr']
  opp_port = data['port']

  game.set_socket(MY_PORT, opp_ip, opp_port)
  game.test_connect()
  socketio.emit('connection_result')


# socketio로 받은 상대방 정보
@socketio.on('opp_data_transfer')
def opp_data_transfer(data):
  global opponent_data
  opponent_data = data['data']


# socketio로 받은 먹이 위치
@socketio.on('set_food_location')
def set_food_loc(data):
  global game
  game.foodPoint = data['foodPoint']


# socketio로 받은 먹이 위치와 상대 점수
@socketio.on('set_food_location_score')
def set_food_loc(data):
  global game
  game.foodPoint = data['foodPoint']
  game.opp_score = data['opp_score']


########################################################################################################################
######################################## MAIN GAME ROUNTING ############################################################
@app.route('/snake')
def snake():
  def generate():
    global opponent_data
    global game
    global gameover_flag

    while True:
      success, img = cap.read()
      img = cv2.flip(img, 1)
      hands, img = detector.findHands(img, flipType=False)

      pointIndex = []

      if hands:
        lmList = hands[0]['lmList']
        pointIndex = lmList[8][0:2]

      img = game.update(img, pointIndex)

      # encode the image as a JPEG string
      _, img_encoded = cv2.imencode('.jpg', img)
      yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + img_encoded.tobytes() + b'\r\n')

      if gameover_flag:  # ^^ 게임 오버 시
        pass

  return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


########################################################################################################################
############################### TEST BED FOR GAME LOGIC DEV ############################################################

# SETTING UP VARIABLES AND FUNCTION FOR BOT
bot_data = {'bot_head_x': 300,
            'bot_head_y': 500,
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

  if bot_data['currentLength'] > 250:
    for i, length in enumerate(bot_data['lengths']):
      bot_data['currentLength'] -= length
      bot_data['lengths'] = bot_data['lengths'][1:]
      bot_data['bot_body_node'] = bot_data['bot_body_node'][1:]

      if bot_data['currentLength'] < 250:
        break


# TEST BED ROUTING
@app.route('/test_bed')
def test_bed():
  def generate():
    global bot_data, game, gameover_flag,sid
    global opponent_data

    game.multi=False
    while True:
      success, img = cap.read()
      img = cv2.flip(img, 1)
      hands, img = detector.findHands(img, flipType=False)

      pointIndex = []

      if hands:
        lmList = hands[0]['lmList']
        pointIndex = lmList[8][0:2]

      bot_data_update()
      opponent_data['opp_body_node']=bot_data["bot_body_node"]
      # print(pointIndex)
      img = game.update(img,pointIndex)

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
