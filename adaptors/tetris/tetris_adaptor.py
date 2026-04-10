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
        gui.PAUSE = 0.01

    def resetinput(self):
        print("Resetting...")
        time.sleep(0.1)
        gui.press("f1")
        gui.press("enter")
        self.total_holes = 0
    
    def stepinput(self, action):

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

        penalty = (change * 10)

        reward = 10 - penalty
        print("total holes: " + str(self.total_holes))
        print("reward: " + str(reward))
        return reward

    def isDone(self):
        frame = self.env.frame > 0
        done = True
        for pixel in frame[0]:
            if not pixel:
                done = False

        return done