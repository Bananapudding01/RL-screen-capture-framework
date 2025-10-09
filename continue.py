from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from main import DinoEnv
import torch
import os
import argparse

if torch.backends.mps.is_available():
    device = "mps"
    print("Using Metal GPU (MPS)")
elif torch.cuda.is_available():
    device = "cuda"
    print("Using CUDA GPU")
else:
    device = "cpu"
    print("Using CPU")

# Parse command line arguments
parser = argparse.ArgumentParser(description='Continue training a Dino PPO model')
parser.add_argument('--model', type=str, default='dino_ppo_final.zip',
                    help='Path to the model to continue training (default: dino_ppo_final.zip)')
parser.add_argument('--timesteps', type=int, default=50000,
                    help='Additional timesteps to train (default: 200000)')
parser.add_argument('--save-name', type=str, default='dino_ppo_continued',
                    help='Name for the saved model (default: dino_ppo_continued)')
args = parser.parse_args()

# Check if model exists
if not os.path.exists(args.model):
    print(f"Error: Model '{args.model}' not found!")
    print("\nAvailable models:")
    for file in os.listdir('.'):
        if file.endswith('.zip'):
            print(f"  - {file}")
    exit(1)

print(f"\nLoading model from: {args.model}")
print("Creating environment...")
env = DinoEnv()

print("Loading existing model...")
model = PPO.load(args.model, env=env, device=device)

# Update learning rate if desired (optional - you can lower it for fine-tuning)
# model.learning_rate = 0.00005  # Uncomment to use a lower learning rate

# Save checkpoints every 10k steps
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path="./checkpoints/",
    name_prefix=f"{args.save_name}"
)

print("\nContinuing training...")
print(f"Training for {args.timesteps:,} additional timesteps")
print("Watch the dino improve! Training will save checkpoints every 10k steps.")
print("You can monitor progress with TensorBoard:")
print("  tensorboard --logdir ./dino_tensorboard/")
print("\nPress Ctrl+C to stop early if needed.\n")

try:
    # Continue training
    model.learn(
        total_timesteps=args.timesteps,
        callback=checkpoint_callback,
        progress_bar=True,
        reset_num_timesteps=False  # Don't reset timestep counter
    )
    
    model.save(args.save_name)
    print(f"\n✓ Training complete! Model saved as '{args.save_name}.zip'")
    
except KeyboardInterrupt:
    print("\n\nTraining interrupted by user.")
    model.save(f"{args.save_name}_interrupted")
    print(f"✓ Progress saved as '{args.save_name}_interrupted.zip'")

print("\nTo test your model, run:")
print(f"  python test.py --model {args.save_name}.zip")