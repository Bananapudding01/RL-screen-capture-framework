import pyautogui
import time
import sys

while True:
    x, y = pyautogui.position()
    pixel = pyautogui.pixel(x, y)
    sys.stdout.write('\033[H')
    sys.stdout.write(f'x = {x}\033[K\ny = {y}\033[K\n{pixel}\033[K\n')
    sys.stdout.flush()
    time.sleep(0.01)