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
import importlib
import platform
OS = platform.system()

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
        self.sct = mss.mss()
        
        # load config file
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            self.game = self.config["game"]
            
            # example path: adaptors.tetris.tetris_adaptor
            path = "adaptors." + self.game + "." + self.game + "_adaptor"
            module = importlib.import_module(path)

        self.game_cords = self.config["game_region"]
        self.shape = self.config["input_shape"]
        self.frame_stack_size = self.shape["frame_stack"]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]), 
            shape=(3,),
            dtype=np.float32
        )

        # example path: adaptors.tetris.tetris_adaptor
        path = "adaptors." + self.config["adaptor"] + "." + self.config["adaptor"] + "_adaptor"
        module = importlib.import_module(path)
        
        self.step_count = 0
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
            
        self.frames.clear()
        for _ in range(self.shape["frame_stack"]):
            self.frames.append(frame)
        
        obs = np.stack(self.frames, axis=-1)
        self.time = time.time()
        self.last_score = 0
        self.step_count = 0
        self.frame_skip = 0
        return obs, {}
    
    def step(self, action):
        self.step_count += 1
        last = time.time()
        reward = 0
 
        # Screen capture
        frame = self._gamecap(self.game_cords, True, self.shape["width"], self.shape["height"])
        self.frames.append(frame)
        obs = np.stack(self.frames, axis=-1)
        
        # Time check
        if time.time() - 29.5 > self.time:
            done = True
        else: 
            done = False
        
        # Safety check
        ########## MAGIC NUMBERS
        x, y = gui.position()
        if x != 1280 or y != 707:
            done = True
        ########## MAGIC NUMBERS
        
        # OCR (every 20 steps)
        # FPS cap
        dt = time.time() - last
        if dt < self.frame_time:
            time.sleep(self.frame_time - dt)
        
        truncated = False
        info = {}
        return obs, reward, done, truncated, info