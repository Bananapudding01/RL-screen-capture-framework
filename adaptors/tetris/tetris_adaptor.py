import numpy as np
import pydirectinput as gui
import time

FRAME = 1/150

class GameAdaptor:
    def __init__(self, env):
        self.env = env
        self.prev_filled = 0
        self.prev_holes = 0
        self.total_lines_cleared = 0
        print("TetrisAdaptor initialized")
        gui.PAUSE = FRAME
        gui.FAILSAFE = False

    def resetinput(self):
        time.sleep(0.1)
        gui.press("f1")
        gui.press("enter")
        self.prev_filled = 0
        self.prev_holes = 0
        self.total_lines_cleared = 0

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

    def _count_holes(self, board):
        holes = 0
        for col in board.T:
            under_block = False
            for cell in col:
                if cell:
                    under_block = True
                elif under_block:
                    holes += 1
        return holes

    def rewardinput(self):
        if self.isDone():
            return -200

        full_frame = self.env.frame > 0
        current_filled = int(np.sum(full_frame))
        cell_delta = current_filled - self.prev_filled

        if cell_delta < 0:
            cell_delta = abs(cell_delta)
            lines_cleared = (cell_delta / 10)
            self.total_lines_cleared += lines_cleared
            line_reward = (cell_delta ** 1.5) * 100
        else:
            line_reward = 0

        self.prev_filled = current_filled

        reward = line_reward + 10
        return reward

    def isDone(self):
        frame = self.env.frame > 0
        for pixel in frame[0][:8]:
            if not pixel:
                return False
        return True