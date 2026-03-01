from stable_baselines3 import PPO
from base_env import BaseEnv
import time
import yaml


with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)
    game = config["game"]

model = PPO.load(game + "_ppo_final.zip")

env = BaseEnv()

print("Running trained model...")

obs, info = env.reset()
episode_reward = 0

for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    
    obs, reward, done, truncated, info = env.step(action)
    episode_reward += reward
    
    if reward > 0:
        print(f"HIT! Reward: {reward}")
    
    if done:
        print(f"Episode reward: {episode_reward}\n")
        episode_reward = 0
        obs, info = env.reset()
        time.sleep(1)