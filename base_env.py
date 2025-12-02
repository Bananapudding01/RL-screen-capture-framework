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

class BaseEnv(gym.Env):

    def _gameover_pixel(self):
        print("checking")
        img = np.array(self.sct.grab({
            "top": self.config["game_over_pixel"]["top"], 
            "left": self.config["game_over_pixel"]["left"],
            "width": 1,
            "height": 1
            }))
        print(img)
        print(self.config["game_over_pixel"]["top"])
        print(self.config["game_over_pixel"]["left"])

        if img[0, 0, 0] == self.config["game_over_pixel"]["RED"]:
            print("red true")
            return True
        else:
            return False



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
        self.action_space = spaces.Discrete(self.config["actions"]["discrete_actions"])
        self.preprocessing = self.config["preprocessing"]
        self.frame_time = 1 / self.preprocessing["fps"]
        self.scorecap_settings = self.config["score_capture"]

        
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
        gui.moveTo(300, 700)
        gui.click()
        gui.press('space')

        test = True
        while test:
            if self._gameover_pixel() == False:
                test = False
            else:
                gui.press('space')

        self.frames.clear()
        for _ in range(self.shape["frame_stack"]):
            self.frames.append(frame)
        
        obs = np.stack(self.frames, axis=-1)

        self.frame_skip = 0
        return obs, {}
    def step(self, action):
        last = time.time()
        reward = 0

        if action == 1:
            gui.press("w")
        elif action == 2:
            gui.press("a")
        elif action == 3:
            gui.press("s")
        elif action == 4:
            gui.press("d")

        frame = self._gamecap(self.game_cords, True, self.shape["width"], self.shape["height"])
        self.frames.append(frame)
        obs = np.stack(self.frames, axis=-1)

        done = self._gameover_pixel()

        # Calculate reward
        if done:
            reward = -100.0 
        else:
            score = self._scorecap()
            if score != "":
                reward = int(score)
            else:
                reward = 0

        # Fps cap
        dt = time.time() - last
        if dt < self.frame_time:
            time.sleep(self.frame_time - dt)

        truncated = False
        info = {}
        return obs, reward, done, truncated, info