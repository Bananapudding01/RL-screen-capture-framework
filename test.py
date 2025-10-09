from stable_baselines3 import PPO
from main import DinoEnv
import time
import tensorboard

model = PPO.load("dino_ppo_final")  # or whatever you named it
env = DinoEnv()

print("Running trained model... Watch the game!")
obs, info = env.reset()

for i in range(1000):  # Run for 1000 steps
    action, _states = model.predict(obs, deterministic=True)
    
    if action == 1:
        print(f"Step {i}: JUMPING")
    else:
        print(f"Step {i}: doing nothing")
    
    obs, reward, done, truncated, info = env.step(action)
    
    if done:
        print(f"Game over! Starting new episode...")
        obs, info = env.reset()

print("Test complete!")