import pydirectinput as gui
import time
import random

time.sleep(2)

while True:
    gui.PAUSE = 0.01
    action = random.randint(0, 39)
    action = 15

    moves = action % 10
    rotation = (action // 10)
    for i in range(rotation):
        gui.press("up") 
    if moves < 5:
        for i in range(moves + 1):
            gui.press("left")
    else:
        for i in range(moves - 4):
            gui.press("right")

    gui.press("down")