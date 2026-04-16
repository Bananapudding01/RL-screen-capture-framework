import numpy as np
import pydirectinput as gui
import time

FRAME = 1 / 150

# ---------------------------------------------------------------------------
# Reward weights — Dellacherie ratios (4:1:1:1:1), scaled down for per-frame RL.
# ---------------------------------------------------------------------------
W_HOLES     = 1.0
W_LANDING   = 0.25
W_ROW_TRANS = 0.25
W_COL_TRANS = 0.25
W_WELLS     = 0.25

SURVIVAL = 1.5
DEATH = -200.0
CLEAR_SETTLE_FRAMES = 2

# Top rows where pieces spawn — excluded from feature computation so the
# falling piece doesn't poison shaping (phantom holes, fake height spikes).
# The agent still SEES these rows in its observation (important for knowing
# which piece it has). Only the reward function ignores them.
SPAWN_ROWS = 4


class GameAdaptor:
    def __init__(self, env):
        self.env = env
        self.prev_filled = 0
        self.prev_holes = 0
        self.prev_max_height = 0
        self.prev_row_trans = 0
        self.prev_col_trans = 0
        self.prev_wells = 0
        self.pending_clear_cells = 0
        self.frames_since_drop = 0
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
        self.prev_max_height = 0
        self.prev_row_trans = 0
        self.prev_col_trans = 0
        self.prev_wells = 0
        self.pending_clear_cells = 0
        self.frames_since_drop = 0
        self.total_lines_cleared = 0

    def stepinput(self, action):
        moves = action % 10
        rotation = action // 10
        for _ in range(rotation):
            gui.press("up")
        if moves < 5:
            for _ in range(moves + 1):
                gui.press("left")
        else:
            for _ in range(moves - 4):
                gui.press("right")
        gui.press("down")
        time.sleep(0.025)

    # ---------------- feature extraction ----------------
    # All functions take `board` which is the CLEAN board (spawn rows stripped).
    # They use board.shape[0] instead of hardcoded 20.

    def _column_heights(self, board):
        rows = board.shape[0]
        heights = np.zeros(10, dtype=np.int32)
        for c in range(10):
            col = board[:, c]
            if col.any():
                heights[c] = rows - int(np.argmax(col))
        return heights

    def _count_holes(self, board):
        holes = 0
        for c in range(10):
            col = board[:, c]
            if not col.any():
                continue
            top = int(np.argmax(col))
            holes += int(np.sum(~col[top:]))
        return holes

    def _row_transitions(self, board):
        padded = np.pad(board, ((0, 0), (1, 1)), constant_values=True)
        return int(np.sum(padded[:, :-1] != padded[:, 1:]))

    def _col_transitions(self, board):
        padded = np.pad(board, ((0, 1), (0, 0)), constant_values=True)
        return int(np.sum(padded[:-1, :] != padded[1:, :]))

    def _count_wells(self, heights):
        wells = 0
        for c in range(10):
            left = heights[c - 1] if c > 0 else 20
            right = heights[c + 1] if c < 9 else 20
            depth = min(left, right) - heights[c]
            if depth >= 2:
                wells += depth
        return wells

    # ---------------- reward ----------------

    def rewardinput(self):
        if self.isDone():
            return DEATH

        full_board = self.env.frame > 0

        # --- fill tracking uses FULL board (spawn piece is ~4 cells every
        # frame so it cancels out in the delta) ---
        current_filled = int(np.sum(full_board))
        cell_delta = current_filled - self.prev_filled
        self.prev_filled = current_filled

        # --- line clear: accumulate during animation, pay out on settle ---
        line_reward = 0.0
        animation_just_ended = False

        if cell_delta < 0:
            self.pending_clear_cells += -cell_delta
            self.frames_since_drop = 0
        elif self.pending_clear_cells > 0:
            self.frames_since_drop += 1
            if self.frames_since_drop >= CLEAR_SETTLE_FRAMES:
                total_drop = self.pending_clear_cells
                approx_lines = max(1, round(total_drop / 10.0))
                self.total_lines_cleared += approx_lines
                line_reward = (total_drop ** 1.5) * 2.0
                if approx_lines >= 4:
                    line_reward *= 2.5
                elif approx_lines == 3:
                    line_reward *= 1.3
                self.pending_clear_cells = 0
                self.frames_since_drop = 0
                animation_just_ended = True

        # --- Dellacherie-style delta shaping ---
        # Strip top SPAWN_ROWS so the falling piece doesn't create phantom
        # holes, fake height spikes, or transition noise.
        shaping_reward = 0.0
        in_animation = self.pending_clear_cells > 0

        if not in_animation:
            clean_board = full_board[SPAWN_ROWS:]  # rows 4-19, shape (16, 10)

            heights   = self._column_heights(clean_board)
            holes     = self._count_holes(clean_board)
            max_h     = int(heights.max())
            row_trans = self._row_transitions(clean_board)
            col_trans = self._col_transitions(clean_board)
            wells     = self._count_wells(heights)

            if not animation_just_ended:
                hole_delta = max(0, holes - self.prev_holes)
                max_delta  = max(0, max_h - self.prev_max_height)
                well_delta = max(0, wells - self.prev_wells)
                row_delta  = row_trans - self.prev_row_trans
                col_delta  = col_trans - self.prev_col_trans

                shaping_reward = (
                    -W_HOLES     * hole_delta
                    -W_LANDING   * max_delta
                    -W_ROW_TRANS * row_delta
                    -W_COL_TRANS * col_delta
                    -W_WELLS     * well_delta
                )

            self.prev_holes = holes
            self.prev_max_height = max_h
            self.prev_row_trans = row_trans
            self.prev_col_trans = col_trans
            self.prev_wells = wells

        return float(line_reward + shaping_reward + SURVIVAL)

    def isDone(self):
        frame = self.env.frame > 0
        for pixel in frame[0][:8]:
            if not pixel:
                return False
        return True