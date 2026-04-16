from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from base_env import BaseEnv
import torch
import yaml
import pydirectinput as gui

gui.PAUSE = 1/150

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

# --- DEBUG: check what the agent actually sees ---
'''
import numpy as np
obs, _ = env.reset()
for i in range(5):
    action = env.action_space.sample()
    obs, r, done, _, _ = env.step(action)
    frame = obs.reshape(20, 10)
    print(f"step {i}: filled={frame.sum()}, reward={r}, done={done}")
    np.savetxt(f"debug_frame_{i}.txt", frame, fmt="%d")
print("Debug frames saved. Check debug_frame_*.txt")
input("Press enter to continue to training (or Ctrl+C to stop)...")
'''
# --- END DEBUG ---

policy = env.policy + "Policy"

print("Creating PPO model with optimized hyperparameters...")
model = PPO(
    policy,
    env,
    verbose=1,
    learning_rate=0.0001,
    n_steps=1024,
    batch_size=64,
    n_epochs=10,
    gamma=0.99, 
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01 ,
    vf_coef=0.5,
    max_grad_norm=0.5,
    tensorboard_log="./" + game + "_tensorboard/",
)

frequency = 20000

# Save checkpoints frequency

checkpoint_callback = CheckpointCallback(
    save_freq=frequency,
    save_path="./checkpoints/",
    name_prefix= game + "_model"
)

print("\nStarting training...")
print("Training will save checkpoints every " + str(frequency) + " steps.")
print("You can monitor progress with TensorBoard:")
print("  tensorboard --logdir ./" + game + "_tensorboard/")
print("\nPress Ctrl+C to stop early if needed.\n")

try:
    model.learn(
        total_timesteps= 1000000,
        callback=[checkpoint_callback],
        progress_bar=True
    )
    
    model.save(game + "_ppo_final")
    print("\n✓ Training complete! Model saved as '" + game + "_ppo_final.zip' ")
    
except KeyboardInterrupt:
    print("\n\nTraining interrupted by user.")
    model.save(game + "_ppo_interrupted")
    print("✓ Progress saved as '" + game + "_ppo_interrupted.zip'")

print("\nTo test your model, run:")
print("  python test.py")
print("\nTo view tensorboard, run: tensorboard --logdir ./" + game + "_tensorboard/")
print("\nThen open: http://localhost:6006")