import numpy as np
import pydirectinput as gui
import time

FRAME = 1/120
FRAME = 1/180

class GameAdaptor:
    def __init__(self, env):
        self.env = env
        self.prev_filled = 0
        self.prev_holes = 0
        print("TetrisAdaptor initialized")
        gui.PAUSE = FRAME

    def resetinput(self):
        print("Resetting...")
        time.sleep(0.1)
        gui.press("f1")
        gui.press("enter")
        self.prev_filled = 0
        self.prev_holes = 0

    def stepinput(self, action):
        moves = action % 10
        rotation = action // 10

        for _ in range(rotation):
            gui.press("up")

        for _ in range(10):
            gui.press("left")

        for _ in range(moves):
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
            return -500

        full_frame = self.env.frame > 0
        current_filled = int(np.sum(full_frame))
        cell_delta = current_filled - self.prev_filled

        new_piece = (cell_delta % 10 == 4)

        lines_cleared = max(0, (4 - cell_delta)) // 10
        line_reward = (lines_cleared ** 1.5) * 500

        hole_penalty = 0
        if new_piece:
            board = full_frame[3:]
            holes = self._count_holes(board)
            hole_penalty = (holes - self.prev_holes) * 3
            self.prev_holes = holes

        self.prev_filled = current_filled

        reward = line_reward - hole_penalty + 10
        print(f"new_piece={new_piece} lines={lines_cleared} hole_penalty={hole_penalty} reward={reward:.1f}")
        return reward

    def isDone(self):
        frame = self.env.frame > 0
        for pixel in frame[0]:
            if not pixel:
                return False
        return True