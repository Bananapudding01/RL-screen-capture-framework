import numpy as np
import pydirectinput as gui
import time

FRAME = 1 / 150

# ---------------------------------------------------------------------------
# Reward weights — based on Dellacherie's classical Tetris features.
#
# Dellacherie used (for one-shot placement scoring):
#   -4 * holes, -1 * wells, -1 * row_trans, -1 * col_trans, -1 * landing_height
# We scale the whole thing down ~4x for per-frame RL shaping, preserving the
# 4:1 hole-dominance ratio. Survival must exceed typical per-step shaping loss
# so the agent can't reward-hack by dying fast.
# ---------------------------------------------------------------------------
W_HOLES      = 1.0   # new holes (asymmetric — only penalize creation)
W_LANDING    = 0.25  # max-height rise (proxy for landing height)
W_ROW_TRANS  = 0.25  # row transitions delta (symmetric — good placements reduce)
W_COL_TRANS  = 0.25  # column transitions delta (symmetric)
W_WELLS      = 0.25  # new wells (asymmetric)

SURVIVAL = 1.5       # per-step survival bonus — average clean play nets positive
DEATH = -200.0       # must dominate max accumulated survival to prevent die-fast
CLEAR_SETTLE_FRAMES = 2  # frames of no-drop before line clear pays out


class GameAdaptor:
    def __init__(self, env):
        self.env = env
        # feature baselines for delta shaping
        self.prev_filled = 0
        self.prev_holes = 0
        self.prev_max_height = 0
        self.prev_row_trans = 0
        self.prev_col_trans = 0
        self.prev_wells = 0
        # line clear animation tracking
        self.pending_clear_cells = 0
        self.frames_since_drop = 0
        # stats
        self.total_lines_cleared = 0
        print("TetrisAdaptor initialized")
        gui.PAUSE = FRAME
        gui.FAILSAFE = False

    def resetinput(self):
        time.sleep(0.1)
        gui.press("f1")
        gui.press("enter")
        # reset EVERY state variable — leaks across episodes cause silent bugs
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
        gui.press("down")  # hard drop (modded ROM)

        time.sleep(0.025) # wait for new block to appear

    # ---------------- feature extraction ----------------

    def _column_heights(self, board):
        heights = np.zeros(10, dtype=np.int32)
        for c in range(10):
            col = board[:, c]
            if col.any():
                heights[c] = 20 - int(np.argmax(col))
        return heights

    def _count_holes(self, board):
        """Empty cells with a filled cell above them in the same column."""
        holes = 0
        for c in range(10):
            col = board[:, c]
            if not col.any():
                continue
            top = int(np.argmax(col))
            holes += int(np.sum(~col[top:]))
        return holes

    def _row_transitions(self, board):
        """Transitions along each row, treating the side walls as filled."""
        padded = np.pad(board, ((0, 0), (1, 1)), constant_values=True)
        return int(np.sum(padded[:, :-1] != padded[:, 1:]))

    def _col_transitions(self, board):
        """Transitions along each column, treating the floor as filled."""
        padded = np.pad(board, ((0, 1), (0, 0)), constant_values=True)
        return int(np.sum(padded[:-1, :] != padded[1:, :]))

    def _count_wells(self, heights):
        """Cumulative depth of all wells (gaps >= 2 deep between columns)."""
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

        board = self.env.frame > 0
        current_filled = int(np.sum(board))
        cell_delta = current_filled - self.prev_filled
        self.prev_filled = current_filled

        # -------- line clear: accumulate during animation, pay out on settle --------
        line_reward = 0.0
        animation_just_ended = False

        if cell_delta < 0:
            # drop frame (cells vanishing during clear animation)
            self.pending_clear_cells += -cell_delta
            self.frames_since_drop = 0
        elif self.pending_clear_cells > 0:
            self.frames_since_drop += 1
            if self.frames_since_drop >= CLEAR_SETTLE_FRAMES:
                total_drop = self.pending_clear_cells
                approx_lines = max(1, round(total_drop / 10.0))
                self.total_lines_cleared += approx_lines

                # superlinear scaling rewards multi-line clears
                line_reward = (total_drop ** 1.5) * 2.0
                # explicit multipliers make tetrises worth the wait
                if approx_lines >= 4:
                    line_reward *= 2.5   # tetris ~ 1265
                elif approx_lines == 3:
                    line_reward *= 1.3   # triple ~ 427

                self.pending_clear_cells = 0
                self.frames_since_drop = 0
                animation_just_ended = True

        # -------- Dellacherie-style delta shaping --------
        # Skip while animation is in flight (features are garbage mid-clear).
        # On the settle frame, refresh baselines WITHOUT computing delta,
        # otherwise the post-clear frame double-pays the line reward via shaping.
        shaping_reward = 0.0
        in_animation = self.pending_clear_cells > 0

        if not in_animation:
            heights = self._column_heights(board)
            holes = self._count_holes(board)
            max_h = int(heights.max())
            row_trans = self._row_transitions(board)
            col_trans = self._col_transitions(board)
            wells = self._count_wells(heights)

            if not animation_just_ended:
                # ASYMMETRIC penalties — only penalize worsening, don't reward the
                # improvements that come "free" from line clears (line_reward pays those)
                hole_delta = max(0, holes - self.prev_holes)
                max_delta  = max(0, max_h - self.prev_max_height)
                well_delta = max(0, wells - self.prev_wells)

                # SYMMETRIC deltas — good placements reduce these naturally, reward that
                row_delta = row_trans - self.prev_row_trans
                col_delta = col_trans - self.prev_col_trans

                shaping_reward = (
                    -W_HOLES     * hole_delta
                    -W_LANDING   * max_delta
                    -W_ROW_TRANS * row_delta
                    -W_COL_TRANS * col_delta
                    -W_WELLS     * well_delta
                )

            # refresh baselines whenever features are computed
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