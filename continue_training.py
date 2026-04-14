from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from base_env import BaseEnv
import torch
import yaml
import sys

#if len(sys.argv) < 2:
    #print("Usage: python continue_training.py <path_to_model>")
    #print("Example: python continue_training.py checkpoints/tetris_model_40000_steps.zip")
    #sys.exit(1)

#model_path = sys.argv[1]
model_path = "tetris_ppo_final.zip"

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

print("\nContinuing training...")
print("Training will save checkpoints every " + str(frequency) + " steps.")
print("You can monitor progress with TensorBoard:")
print("  tensorboard --logdir ./" + game + "_tensorboard/")
print("\nPress Ctrl+C to stop early if needed.\n")

try:
    model.learn(
        total_timesteps=1000000,
        callback=checkpoint_callback,
        progress_bar=True,
        reset_num_timesteps=False,
    )

    model.save(game + "_ppo_final")
    print("\n✓ Training complete! Model saved as '" + game + "_ppo_final.zip'")

except KeyboardInterrupt:
    print("\n\nTraining interrupted by user.")
    model.save(game + "_ppo_interrupted")
    print("✓ Progress saved as '" + game + "_ppo_interrupted.zip'")

print("\nTo test your model, run:")
print("  python test.py")
print("\nTo view tensorboard, run: tensorboard --logdir ./" + game + "_tensorboard/")
print("\nThen open: http://localhost:6006")