import numpy as np
import pytesseract
from PIL import Image

class tetrisAdaptor:
    def __init__(self, env):
        print("TetrisAdaptor initialized")

    def _scorecap(self):
        game_capture = np.array(self.sct.grab(
            self.scorecap_settings
            ))
        img = Image.fromarray(game_capture)
        score_text = pytesseract.image_to_string(img, config='--psm 7 -c tessedit_char_whitelist=0123456789')
        return score_text

    def resetinput(self):
        print("reset input")
    
    def stepinput(self):
        print("step input")

    def rewardinput(self):
        print("reward input")