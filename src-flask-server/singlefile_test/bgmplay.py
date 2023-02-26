import time

import simpleaudio as sa
import threading
import signal

########################################## SETTING BGM PATH ############################################################
bgm_path = '../static/bgm/main.wav'
vfx_1_path = '../static/bgm/curSelect.wav'
vfx_2_path = '../static/bgm/eatFood.wav'
vfx_3_path = '../static/bgm/boost.wav'
vfx_4_path = '../static/bgm/gameOver.wav'
vfx_5_path = '../static/bgm/stageWin.wav'
########################################################################################################################

# GOLOBAL VARIABLE FOR BGM PLAY OBJECT
bgm_play_obj = None

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

# Start playing the BGM
bgm_thread.start()

# Simple thread testing :D
for i in range(15):
  sfx_thread = threading.Thread(target=play_selected_sfx, args=(vfx_1_path,))
  sfx_thread.start()
  time.sleep(0.1)

# Loop to play sound effects selected by the user
while True:
  select_sfx = input("Select SFX to play alongside with BGM (1-5, 9 to stop BGM): ")
  if select_sfx == '0':
    bgm_thread_re = threading.Thread(target=play_bgm)
    bgm_thread_re.start()
  elif select_sfx == '9':
    stop_bgm()
  elif select_sfx == '1':
    sfx_thread = threading.Thread(target=play_selected_sfx, args=(vfx_1_path,))
    sfx_thread.start()
  elif select_sfx == '2':
    sfx_thread = threading.Thread(target=play_selected_sfx, args=(vfx_2_path,))
    sfx_thread.start()
  elif select_sfx == '3':
    sfx_thread = threading.Thread(target=play_selected_sfx, args=(vfx_3_path,))
    sfx_thread.start()
  elif select_sfx == '4':
    sfx_thread = threading.Thread(target=play_selected_sfx, args=(vfx_4_path,))
    sfx_thread.start()
  elif select_sfx == '5':
    sfx_thread = threading.Thread(target=play_selected_sfx, args=(vfx_5_path,))
    sfx_thread.start()
  else:
    print("말좀들으세요")
