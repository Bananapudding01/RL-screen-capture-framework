import pyautogui as gui
import time 

while True:
    x, y = gui.position()
    print(x, y)
    print(gui.pixel(x-5, y-5))
    time.sleep(0.05)