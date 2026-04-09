# RL Screen Capture Framework

A reinforcement learning framework that trains agents to play games using screen capture. Instead of hooking into game internals, it captures a region of your screen, preprocesses the image into observations, and sends keyboard/mouse inputs back to the game — making it compatible with virtually any game or application.

Built on [Gymnasium](https://gymnasium.farama.org/) and [Stable Baselines3](https://stable-baselines3.readthedocs.io/), with PPO as the default training algorithm.

**Status:** Early-stage / work in progress. The core framework is functional but individual game adaptors are still being refined.

## How It Works

```
Screen Capture  -->  Preprocessing  -->  RL Agent (PPO)  -->  Keyboard/Mouse Input
 (mss)              (OpenCV)            (SB3)                (pyautogui)
```

1. **Capture** — Uses [mss](https://github.com/BoboTiG/python-mss) to grab a region of the screen relative to a target application window.
2. **Preprocess** — Optionally converts to grayscale, applies black/white thresholding, and resizes to the configured input dimensions.
3. **Observe** — Stacks frames (if configured) and feeds them to the agent as a CNN or MLP observation.
4. **Act** — The agent selects an action, which the game-specific adaptor translates into keyboard/mouse inputs.
5. **Reward** — The adaptor computes a reward signal (from pixel analysis, OCR score reading, or custom logic).
6. **Repeat** — The loop runs at a configurable FPS.

## Project Structure

```
.
├── config.yaml                 # Top-level config — sets which game to use
├── base_env.py                 # Core Gymnasium environment
├── train_ppo.py                # Training script (PPO via Stable Baselines3)
├── test.py                     # Inference script — runs a trained model
├── set_gamecap.py              # Interactive tool to calibrate game capture region
├── set_scorecap.py             # Interactive tool to calibrate score capture region
├── getmouse.py                 # Mouse position & pixel color debug tool (Windows)
├── pixelcolor.py               # Pixel color inspector
├── input_test.py               # Input system test
├── requirements-mac.txt        # Dependencies for macOS
├── requirements-win.txt        # Dependencies for Windows
└── adaptors/
    ├── tetris/                 # NES Tetris (via Mesen emulator)
    │   ├── tetris_config.yaml
    │   └── tetris_adaptor.py
    ├── dino/                   # Chrome Dino game
    │   ├── dino_config.yaml
    │   └── dino_adaptor.py
    └── test/                   # Stub adaptor for development/testing
        ├── test_config.yaml
        └── test_adaptor.py
```

## Prerequisites

- **Python 3.10+**
- **Tesseract OCR** — Required for score reading via OCR.
  - macOS: `brew install tesseract`
  - Windows: Download from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) and add to PATH
  - Linux: `sudo apt install tesseract-ocr`
- **The target game/application** must be running in a visible window (not minimized or fullscreen on a separate desktop).

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Bananapudding01/RL-screen-capture-framework.git
   cd RL-screen-capture-framework
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate        # macOS/Linux
   venv\Scripts\activate           # Windows
   ```

3. Install dependencies:

   **macOS (Metal/MPS GPU support):**
   ```bash
   pip install -r requirements-mac.txt
   ```

   **Windows (CUDA GPU support):**
   ```bash
   # Install PyTorch with CUDA first:
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   # Then install the rest:
   pip install -r requirements-win.txt
   ```

   > **Note:** On Windows, the Tetris adaptor uses `pydirectinput` for input. On macOS, `pyautogui` is used. The Dino adaptor uses `pyautogui` on all platforms.

## Quick Start

### 1. Set the game

Edit `config.yaml` to choose your game:

```yaml
game: "tetris"    # or "dino", "test"
```

This determines which adaptor folder under `adaptors/` is loaded.

### 2. Calibrate the capture region

Open the target game window, then run:

```bash
python set_gamecap.py
```

This launches an interactive tool with a live preview window. It has two stages:

- **Stage 1 — Capture region:** Adjust `top`, `left`, `width`, `height` to frame the game area relative to the application window.
- **Stage 2 — Preprocessing:** Adjust output `height`, `width`, toggle `gray`/`blackwhite`, and set `threshold`.

Type `next` to advance between stages. The settings are saved to the adaptor's config YAML.

To calibrate the score capture region (used for OCR-based score reading):

```bash
python set_scorecap.py
```

### 3. Train

```bash
python train_ppo.py
```

This will:
- Auto-detect your GPU (Metal/MPS on Mac, CUDA on Windows, or CPU fallback)
- Create the environment and load the selected adaptor
- Train a PPO agent and save checkpoints to `./checkpoints/` every 20,000 steps
- Save the final model as `{game}_ppo_final.zip`
- Log training metrics to `{game}_tensorboard/`

Monitor training with TensorBoard:

```bash
tensorboard --logdir ./{game}_tensorboard/
# Then open http://localhost:6006
```

Press `Ctrl+C` to stop early — progress is saved as `{game}_ppo_interrupted.zip`.

### 4. Test a trained model

```bash
python test.py
```

Loads `{game}_ppo_final.zip` and runs 1000 steps of deterministic inference, printing rewards as they come.

## Adaptor System

Each game has a pluggable **adaptor** — a Python class called `GameAdaptor` that defines how the agent interacts with that specific game. Adaptors live in `adaptors/{game}/` alongside a YAML config file.

### Creating a New Adaptor

1. Create a folder: `adaptors/mygame/`
2. Create `mygame_config.yaml`:

   ```yaml
   application_name: "Window Title of Your Game"
   policy: Cnn            # or Mlp
   actions:
     action_space: discrete
     discrete_actions: 4  # number of possible actions
   game_region:
     top: 0               # offset from window top
     left: 0              # offset from window left
     width: 400
     height: 300
   input_shape:
     height: 60           # observation height (pixels)
     width: 80            # observation width (pixels)
     frame_stack: 1       # number of stacked frames
   preprocessing:
     grayscale: true
     blackwhite: false
     threshold: 128
     fps: 30
   score_capture:
     top: 0
     left: 0
     width: 100
     height: 40
   ```

3. Create `mygame_adaptor.py`:

   ```python
   class GameAdaptor:
       def __init__(self, env):
           self.env = env

       def resetinput(self):
           """Send inputs to restart the game."""
           pass

       def stepinput(self, action):
           """Translate the agent's action into game inputs."""
           pass

       def rewardinput(self):
           """Compute and return the reward for the current step."""
           return 0

       def isDone(self):
           """Return True if the episode is over."""
           return False
   ```

4. Set `config.yaml` to `game: "mygame"` and run `set_gamecap.py` to calibrate.

### Included Adaptors

| Adaptor | Game | Policy | Actions | Notes |
|---------|------|--------|---------|-------|
| `tetris` | NES Tetris (Mesen emulator) | MLP | 40 discrete (4 rotations x 10 columns) | Reward based on hole analysis; 20x10 binary grid |
| `dino` | Chrome Dino (`chrome://dino`) | CNN | 4 discrete (up/down combos) | Pixel-based game-over detection; grayscale input |
| `test` | None (stub) | MLP | 4 discrete | Always returns reward=1, never done; for development |

## Configuration Reference

### `config.yaml`
| Field | Description |
|-------|-------------|
| `game` | Name of the adaptor to load (must match a folder in `adaptors/`) |

### Adaptor Config (`{game}_config.yaml`)
| Field | Description |
|-------|-------------|
| `application_name` | Window title of the target application |
| `policy` | `Cnn` (image-based) or `Mlp` (flattened vector) |
| `actions.action_space` | `discrete` or `continuous` |
| `actions.discrete_actions` | Number of discrete actions |
| `actions.continuous_low/high` | Bounds for continuous action spaces |
| `game_region` | Screen capture offset relative to the window (`top`, `left`, `width`, `height`) |
| `input_shape` | Observation dimensions (`height`, `width`, `frame_stack`) |
| `preprocessing` | Image processing toggles (`grayscale`, `blackwhite`, `threshold`, `fps`) |
| `score_capture` | Region for OCR score reading (`top`, `left`, `width`, `height`) |

## Utility Scripts

| Script | Platform | Description |
|--------|----------|-------------|
| `set_gamecap.py` | Windows | Interactive game region + preprocessing calibration with live preview |
| `set_scorecap.py` | All | Interactive score region calibration with live preview |
| `getmouse.py` | Windows | Prints mouse position and pixel color continuously |
| `pixelcolor.py` | All | Prints cursor position and pixel color |
| `input_test.py` | All | Tests pyautogui keyboard input |

## Platform Notes

- **macOS:** Uses `pywinctl` for window management and Metal/MPS for GPU acceleration. Grant accessibility/screen recording permissions when prompted.
- **Windows:** Uses `pygetwindow` for window management and CUDA for GPU acceleration. The Tetris adaptor uses `pydirectinput` for lower-latency input. `set_gamecap.py` uses DPI awareness via `ctypes.windll`.
- **Linux:** Should work with `pywinctl` (same as macOS path), but is not actively tested.

## License

See repository for license information.