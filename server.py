from copyreg import pickle
import datetime
import time, random
from flask import Flask, render_template, Response, request, redirect, url_for, session
from flask_socketio import SocketIO, emit, join_room
import uuid
from engineio.payload import Payload
from socket import *

Payload.max_decode_packets = 200

app = Flask(__name__)
app.config['SECRET_KEY'] = "roomfitisdead"

socketio = SocketIO(app, cors_allowed_origins='*')

waiting_players = []
room_of_players = {}
players_in_room = {}
address = {}
last_created_room = ""

@app.route("/")
def index():
    return render_template("servertime.html")

@socketio.on('connect')
def test_connect():
    ip_addr = request.remote_addr
    port = request.environ['REMOTE_PORT']
    print(f'Client connected: {ip_addr}:{port}')

@socketio.on('disconnect')
def test_disconnect():
    ip_addr = request.remote_addr
    port = request.environ['REMOTE_PORT']
    print(f'Client disconnected: {ip_addr}:{port}')

@socketio.on('room_disconnect')
def room_disconnect(data):
    room_id = data['room_id']
    sid = data['sid']
    print(f'room_id = {room_id} sid = {sid}')
    global room_of_players
    room_of_players = {k: v for k, v in room_of_players.items() if v != room_id}
    print(room_of_players)
    print('user left room')

@socketio.on('gameover_to_server')
def gameover_to_server(data):
    sid = data['sid']

    emit("gameover_to_clients", {'sid' : sid}, broadcast=True, include_self=False)
    print('gameover to clients')

@socketio.on('foodEat_to_server')
def foodEat_to_server(data):
    foodPoint=[]
    # print(f'foodEat_to_server Function Called!!!')
    if data['foodEat']:
        foodPoint=random.randint(100, 1000), random.randint(100, 600)
        emit("foodPoint_from_server", {'foodPoint' : foodPoint}, broadcast=True)
    # print('reposition food')

@socketio.on('join')
def handle_join():
    global last_created_room
    global players_in_room
    if len(waiting_players) == 0:
        waiting_players.append(request.sid)
        last_created_room = str(uuid.uuid4())

        join_room(last_created_room)
        room_of_players[request.sid] = last_created_room
        emit('waiting', {'room_id' : last_created_room, 'sid' : request.sid}, to=last_created_room)
    else:
        host_sid = waiting_players.pop()
        room_id = room_of_players[host_sid]
        join_room(room_id)

        room_of_players[request.sid] = room_id
        players_in_room[room_id] = 0

        last_created_room = ""
        print(room_of_players)
        emit('matched', {'room_id' : room_id, 'sid' : request.sid}, to=room_id)
        emit('start-game', {'room_id' : room_id, 'sid' : request.sid}, to=request.sid)
        emit('start-game', {'room_id' : room_id, 'sid' : host_sid}, to=host_sid)

@socketio.on('send_data')
def send_data(data):
    head_x = data['head_x']
    head_y = data['head_y']
    body_node = data['body_node']
    score = data['score']
    room_id = data['room_id']
    sid = data['sid']

    # print(head_x, head_y, score, room_id, sid)
    # print(f'head_x, head_y, score, room_id, sid')
    emit('opp_data', {'opp_head_x' : head_x, 'opp_head_y' : head_y, 'opp_body_node' : body_node, 'opp_score' : score, 'opp_room_id' : room_id, 'opp_sid' : sid}, broadcast=True, include_self=False)
    # emit('opp_data', {'opp_head_x' : head_x, 'opp_head_y' : head_y, 'opp_body_node' : body_node, 'opp_score' : score, 'opp_room_id' : room_id, 'opp_sid' : sid}, broadcast=True)

@socketio.on('send_data_bot')
def send_data_bot(data):
    head_x = data['head_x']
    head_y = data['head_y']
    body_node = data['body_node']
    score = data['score']
    room_id = data['room_id']
    sid = data['sid']

    # print(head_x, head_y, score, room_id, sid)
    emit('bot_data', {'bot_head_x' : head_x, 'bot_head_y' : head_y, 'bot_body_node' : body_node, 'bot_score' : score, 'bot_room_id' : room_id}, broadcast=True, include_self=False)
    # emit('opp_data', {'opp_head_x' : head_x, 'opp_head_y' : head_y, 'opp_body_node' : body_node, 'opp_score' : score, 'opp_room_id' : room_id, 'opp_sid' : sid}, broadcast=True)


@socketio.on('get_time')
def get_time():
    while True:
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        socketio.emit('time', {'time': current_time})
        socketio.sleep(1)

# 서버에서 현재 자신의 포트 받아오기
# < sock.bind() 작업에서 포트 번호 지정을 위해 필요 >
# < index -> snake 페이지 변환하면서 포트 변경됨 > => 매칭 시 그때 포트를 받은 후 연결
@socketio.on('my_port')
def my_port(data):
    global players_in_room
    global address
    ip_addr = request.remote_addr
    port = request.environ['REMOTE_PORT']
    room_id = data['room_id']

    join_room(room_id)
    players_in_room[room_id] += 1
    emit('my_port', {'my_port':port})

    if players_in_room[room_id] == 2:
        emit('opponent_address', {'ip_addr' : ip_addr, 'port' : port}, broadcast=True, include_self=False, room=room_id)
        emit('opponent_address', {'ip_addr' : address[room_id][0], 'port' : address[room_id][1]})
    else:
        address[room_id] = [ip_addr, port]


if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0', port=80, debug=True)
