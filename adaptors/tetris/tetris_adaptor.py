import numpy as np
import pytesseract
from PIL import Image
import pyautogui as gui

class GameAdaptor:
    def __init__(self, env):
        print("TetrisAdaptor initialized")

    def resetinput(self):
        print("reset input")
    
    def stepinput(self, action):

        gui.PAUSE = 0

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

        for i in range(20):
            gui.press("down")

    def rewardinput(self):
        frame = self.frame > 0



        

    def isDone(self):
        print("is done")
        return False