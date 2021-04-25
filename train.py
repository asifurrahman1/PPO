import argparse
import gym
from ppo import PPO

def run(args):
  env = gym.make('Pendulum-v0')
  agent = PPO(env,args)
  agent.learn()

if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('--total_num_steps', type=int, default=10**6)
  p.add_argument('--steps_per_batch', type=int, default=4800)
  p.add_argument('--steps_per_episode', type=int, default=1600)
  p.add_argument('--gamma', type=float, default=0.95)
  p.add_argument('--epoch_per_iteration', type=int, default=5)
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--save_frequency', type=int, default=10)
  p.add_argument('--path', type=str, default="/gdrive/MyDrive/Colab Notebooks/PPO_data")
  args = p.parse_args()
  run(args)