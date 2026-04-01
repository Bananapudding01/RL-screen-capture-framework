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
import pygetwindow as getwindow
OS = platform.system()

class BaseEnv(gym.Env):

    def _gamecap(self, cords, grayscaled, sizex, sizey, blackwhite, threshold, application):
        window = getwindow.getWindowsWithTitle(application)[0]

        game_cords = {
            'left':   window.left + cords['left'],
            'top':    window.top  + cords['top'],
            'width':  cords['width'],
            'height': cords['height']
        }

        game_capture = np.array(self.sct.grab(game_cords))

        if grayscaled == True: game_capture = cv2.cvtColor(game_capture, cv2.COLOR_BGR2GRAY)
        if blackwhite == True: _, game_capture = cv2.threshold(game_capture, threshold, 255, cv2.THRESH_BINARY)
        if sizex > 0: game_capture = cv2.resize(game_capture, (sizex, sizey), interpolation=cv2.INTER_AREA)
        if blackwhite == True: _, game_capture = cv2.threshold(game_capture, threshold, 255, cv2.THRESH_BINARY)
        return game_capture
        
    def _scorecap(self):
        game_capture = np.array(self.sct.grab(self.scorecap_settings))
        img = Image.fromarray(game_capture)
        score_text = pytesseract.image_to_string(img, config='--psm 7 -c tessedit_char_whitelist=0123456789')
        return score_text
    
    def __init__(self, config_path="config.yaml"):
        super(BaseEnv, self).__init__()
        self.sct = mss.mss()
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            self.game = self.config["game"]

        adaptor_config_path = "adaptors/" + self.game + "/" + self.game + "_config.yaml"
        with open(adaptor_config_path, 'r') as f:
            self.adaptor_config = yaml.safe_load(f)

            self.policy = self.adaptor_config["policy"]
            self.application = self.adaptor_config["application_name"]
            self.action_space_type = self.adaptor_config["actions"]["action_space"]

            if self.action_space_type == "discrete":
                self.action_space = spaces.Discrete(self.adaptor_config["actions"]["discrete_actions"])
            elif self.action_space_type == "continuous":
                self.action_space = spaces.Box(
                    low=np.array(self.adaptor_config["actions"]["continuous_low"]),
                    high=np.array(self.adaptor_config["actions"]["continuous_high"]),
                    shape=(len(self.adaptor_config["actions"]["continuous_low"]),),
                    dtype=np.float32
                )

            self.game_offset = self.adaptor_config["game_region"]
            self.shape = self.adaptor_config["input_shape"]
            self.frame_stack_size = self.shape["frame_stack"]

        self.step_count = 0
        self.last_score = 0
        self.preprocessing = self.adaptor_config["preprocessing"]
        self.frame_time = 1 / self.preprocessing["fps"]
        self.scorecap_settings = self.adaptor_config["score_capture"]

        if self.policy == "Cnn":
            self.observation_space = spaces.Box(
                low=0, 
                high=255,
                shape=(self.shape["height"], self.shape["width"], self.shape["frame_stack"]),
                dtype=np.uint8
            )
        elif self.policy == "Mlp":
            self.observation_space = spaces.Box(
                low=0, 
                high=255,
                shape=(self.shape["height"] * self.shape["width"] * self.shape["frame_stack"],),
                dtype=np.uint8
            )
        else:
            raise ValueError("Unsupported policy type in adaptor config: " + self.policy)
        
        self.frames = deque(maxlen=self.shape["frame_stack"])
        self.time = time.time()

        path = "adaptors." + self.game + "." + self.game + "_adaptor"
        module = importlib.import_module(path)
        self.adaptor = module.GameAdaptor(self)
    
    def reset(self, seed=None, options=None):
        print("Resetting...")
        frame = self._gamecap(
            self.game_offset, 
            self.preprocessing["grayscale"], 
            self.shape["width"], 
            self.shape["height"],
            self.preprocessing["blackwhite"],
            self.preprocessing["threshold"],
            self.application
        )
        
        super().reset(seed=seed)
        self.adaptor.resetinput()
        self.frames.clear()

        for _ in range(self.shape["frame_stack"]):
            self.frames.append(frame)
        
        obs = np.stack(self.frames, axis=-1)
        if self.policy == "Mlp":
            obs = obs.flatten()

        self.time = time.time()
        self.last_score = 0
        self.step_count = 0
        self.frame_skip = 0

        return obs, {}
    
    def step(self, action):
        self.step_count += 1
        last = time.time()

        self.adaptor.stepinput(action)

        frame = self._gamecap(
            self.game_offset, 
            self.preprocessing["grayscale"], 
            self.shape["width"], 
            self.shape["height"],
            self.preprocessing["blackwhite"],
            self.preprocessing["threshold"],
            self.application
        )
        self.frames.append(frame)
        obs = np.stack(self.frames, axis=-1)
        if self.policy == "Mlp":
            obs = obs.flatten()

        done = self.adaptor.isDone()
        reward = self.adaptor.rewardinput()

        dt = time.time() - last
        if dt < self.frame_time:
            time.sleep(self.frame_time - dt)
        
        truncated = False
        info = {}
        return obs, reward, done, truncated, info