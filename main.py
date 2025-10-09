import mss
import time
import cv2 
import numpy as np
from PIL import Image
import pytesseract
import gymnasium as gym
from gymnasium import spaces
import pyautogui as gui
from collections import deque

game_cords = {"top": 300, "left": 310, "width": 300, "height": 60}
score_cords = {"top": 267, "left": 226, "width": 90, "height": 18}

class DinoEnv(gym.Env):

    def _gamecap(self, cords, grayscaled, sizex, sizey):
        game_capture = np.array(self.sct.grab(cords))
        if grayscaled == True:
            game_capture = cv2.cvtColor(game_capture, cv2.COLOR_BGR2GRAY)
        if sizex > 0:
            game_capture = cv2.resize(game_capture, (sizex, sizey), interpolation=cv2.INTER_AREA)

        return game_capture
        
    def _scorecap(self, cords):
        game_capture = np.array(self.sct.grab(cords))
        img = Image.fromarray(game_capture)
        score_text = pytesseract.image_to_string(img)#, config='--psm 7 -c tessedit_char_whitelist=0123456789')
        return score_text

    def __init__(self):
        super(DinoEnv, self).__init__()

        gui.PAUSE = 0.0

        self.sct = mss.mss()

        self.frame_stack_size = 4
        
        self.action_space = spaces.Discrete(2)  # 0=nothing, 1=jump
        
        self.observation_space = spaces.Box(
            low=0, 
            high=255,
            shape=(60, 75, 4),  # grayscaled, 4 frames
            dtype=np.uint8
        )
        self.frames = deque(maxlen=4)
    
    def reset(self, seed=None, options=None):

        print("Resetting...")
        frame = self._gamecap(game_cords, True, 75, 60)
        super().reset(seed=seed)
        gui.moveTo(300, 400)
        gui.click()
        gui.press('space')

        test = True
        while test:
            cords = {"top": 310, "left": 330, "width": 1, "height": 1}
            img = np.array(self.sct.grab(cords))
            if img[0, 0, 0] != 83:
                test = False
            else:
                gui.press('space')
        

        self.frames.clear()
        for _ in range(4):
            self.frames.append(frame)
        
        obs = np.stack(self.frames, axis=-1)

        self.frame_skip = 0
        return obs, {}
    
    def step(self, action):
        last = time.time()
        reward = 0

        if action == 1:
            gui.press("space")

        frame = self._gamecap(game_cords, True, 75, 60)
        self.frames.append(frame)
        obs = np.stack(self.frames, axis=-1)

        cords = {"top": 310, "left": 330, "width": 1, "height": 1}
        img = np.array(self.sct.grab(cords))
        if img[0, 0, 0] == 83:
            done = True
        else:
            done = False

        # Calculate reward
        if done:
            reward = -100.0 
        else:
            reward = 1

        # Fps cap
        dt = time.time() - last
        if dt < 0.067:
            time.sleep(0.067 - dt)

        truncated = False
        info = {}
        return obs, reward, done, truncated, info