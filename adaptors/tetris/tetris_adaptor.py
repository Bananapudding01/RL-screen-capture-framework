import numpy as np
import pytesseract
from PIL import Image
import pydirectinput as gui
import time

class GameAdaptor:
    def __init__(self, env):
        self.total_holes = 0
        self.env = env
        print("TetrisAdaptor initialized")

    def resetinput(self):
        gui.press("space")
        gui.press("space")
    
    def stepinput(self, action):

        gui.PAUSE = 0.00

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
        frame = self.env.frame > 0
        
        #calculating number of holes

        totalholes = 0

        done = True
        for pixel in frame[0]:
            if not pixel:
                done = False

        if done == True:
            return -200


        for column in frame.T:
            holes = 0
            under = False
            
            for block in column:
                if not under:
                    if block:
                        under = True
                else:
                    if not block:
                        holes += 1

            totalholes += holes
        
        change = totalholes - self.total_holes
        self.total_holes = totalholes

        penalty = (change ** 2) * 20
        if change > 0: penalty = penalty * -1

        reward = 10 - penalty
        return reward

    def isDone(self):
        frame = self.env.frame > 0
        done = True
        for pixel in frame[0]:
            if not pixel:
                done = False

        if done == True:
            return -200