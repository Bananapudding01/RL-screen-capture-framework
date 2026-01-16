import mss
import time
import cv2 
import gymnasium as gym
from gymnasium import spaces
import pyautogui as gui
from PIL import Image
import pytesseract
import numpy as np
from collections import deque
import yaml
import pydirectinput as directinput
class BaseEnv(gym.Env):
    def _gamecap(self, cords, grayscaled, sizex, sizey):
        game_capture = np.array(self.sct.grab(cords))
        if grayscaled == True:
            game_capture = cv2.cvtColor(game_capture, cv2.COLOR_BGR2GRAY)
        if sizex > 0:
            game_capture = cv2.resize(game_capture, (sizex, sizey), interpolation=cv2.INTER_AREA)
        return game_capture
        
    def _scorecap(self):
        game_capture = np.array(self.sct.grab(
            self.scorecap_settings
            ))
        img = Image.fromarray(game_capture)
        score_text = pytesseract.image_to_string(img, config='--psm 7 -c tessedit_char_whitelist=0123456789')
        return score_text
    def __init__(self, config_path="config.yaml"):
        super(BaseEnv, self).__init__()
        gui.PAUSE = 0.0
        self.sct = mss.mss()
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.game_cords = self.config["game_region"]
        self.shape = self.config["input_shape"]
        self.frame_stack_size = self.shape["frame_stack"]
        # **CHANGED: Only 2D movement (x, y), no click decision**
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0, 
            shape=(2,),  # Just [x, y]
            dtype=np.float32
        )
        self.last_score = 0
        self.preprocessing = self.config["preprocessing"]
        self.frame_time = 1 / self.preprocessing["fps"]
        self.scorecap_settings = self.config["score_capture"]
        self.time = time.time()
        
        self.observation_space = spaces.Box(
            low=0, 
            high=255,
            shape=(self.shape["height"], self.shape["width"], self.shape["frame_stack"]),
            dtype=np.uint8
        )
        self.frames = deque(maxlen=self.shape["frame_stack"])
    
    def reset(self, seed=None, options=None):
        print("Resetting...")
        frame = self._gamecap(
            self.game_cords, 
            True, 
            self.shape["width"], 
            self.shape["height"]
            )
        super().reset(seed=seed)
        while True:
            gui.moveTo(600, 1000)
            time.sleep(0.5)
            gui.click()
            if gui.position() == (1280, 707):
                break
            
        self.frames.clear()
        for _ in range(self.shape["frame_stack"]):
            self.frames.append(frame)
        
        obs = np.stack(self.frames, axis=-1)
        self.time = time.time()
        self.last_score = 0
        self.frame_skip = 0
        return obs, {}
    def step(self, action):
        last = time.time()
        reward = 0
        
        
        frame = self._gamecap(self.game_cords, True, self.shape["width"], self.shape["height"])
        self.frames.append(frame)
        obs = np.stack(self.frames, axis=-1)

        #time check
        if time.time() - 29.5 > self.time:
            done = True
        else: 
            done = False

        #safety check
        x,y = gui.position()
        if x != 1280:
            if y != 707:
                done = True

        # **CHANGED: Only handle x and y movement (2D action)**
        max_movement = 350
        dx = int(action[0] * max_movement)  # -500 to +500 pixels
        dy = int(action[1] * max_movement)  # -500 to +500 pixels
        directinput.moveRel(dx, dy, relative=True)
        
        # **CHANGED: Automatically click every step**
        gui.click()
        
        # Calculate reward
        
        score = self._scorecap()
        if score != "":
            reward = (int(score) - self.last_score) * 100
            self.last_score = int(score)
        else:
            reward = 0
        # Fps cap
        dt = time.time() - last
        if dt < self.frame_time:
            time.sleep(self.frame_time - dt)
        truncated = False
        info = {}
        print("reward = " + str(reward))
        print("score = " + str(score))
        return obs, reward, done, truncated, info