from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from base_env import BaseEnv
import torch
import yaml
import pydirectinput as gui
import time

model_path = "tetris_ppo_interrupted.zip"

with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)
    game = config["game"]

if torch.backends.mps.is_available():
    device = "mps"
    print("Using Metal GPU (MPS)")
elif torch.cuda.is_available():
    device = "cuda"
    print("Using CUDA GPU")
else:
    device = "cpu"
    print("Using CPU")

class PauseCallback(BaseCallback):
    def _on_step(self):
        if hasattr(self.training_env.envs[0], 'adaptor'):
            adaptor = self.training_env.envs[0].adaptor
            self.logger.record("custom/lines_cleared", adaptor.total_lines_cleared)
        return True
    def _on_rollout_end(self):
        gui.press("f1")
    def _on_rollout_start(self):
        gui.press("f1")
        time.sleep(0.1)
        gui.press("f1")

print("Creating environment...")
env = BaseEnv()

print(f"Loading model from: {model_path}")
model = PPO.load(
    model_path,
    env=env,
    device=device,
    tensorboard_log="./" + game + "_tensorboard/",
)

frequency = 20000
checkpoint_callback = CheckpointCallback(
    save_freq=frequency,
    save_path="./checkpoints/",
    name_prefix=game + "_model"
)
pause_callback = PauseCallback()

print("\nContinuing training...")
try:
    model.learn(
        total_timesteps=1000000,
        callback=[checkpoint_callback, pause_callback],
        progress_bar=True,
        reset_num_timesteps=False,
    )
    model.save(game + "_ppo_final")
    print("\n✓ Training complete! Model saved as '" + game + "_ppo_final.zip'")
except KeyboardInterrupt:
    print("\n\nTraining interrupted by user.")
    model.save(game + "_ppo_interrupted")
    print("✓ Progress saved as '" + game + "_ppo_interrupted.zip'")