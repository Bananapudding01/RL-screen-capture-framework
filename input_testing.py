import pydirectinput as gui
import time
import random

FRAME = 1/30

gui.PAUSE = 1/60

time.sleep(5)

ACTION_NAMES = ["nothing", "left", "right", "rotate"]

while True:
    action = random.randint(0, 3)
    print(f"Action: {ACTION_NAMES[action]}")

    if action == 0:
        pass
    elif action == 1:
        gui.press("left")
    elif action == 2:
        gui.press("right")
    elif action == 3:
        gui.press("up")
    
    #time.sleep(FRAME)