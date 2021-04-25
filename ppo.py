import torch
import os
import numpy as np
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.optim import Adam
from network import FFNetwork
import matplotlib.pyplot as plt

class PPO:
    def _init_hyperparameters(self, env, args):
      self.obs_dim = env.observation_space.shape[0]
      self.act_dim = env.action_space.shape[0]
      print(self.obs_dim)
      print(self.act_dim)
      self.total_step = args.total_num_steps
      self.steps_per_batch = args.steps_per_batch
      self.steps_per_episode = args.steps_per_episode
      self.gamma = args.gamma
      self.epoch_per_iteration = args.epoch_per_iteration
      self.clip = 0.2
      self.lr = 0.005
      self.save_freq = args.save_frequency
      self.path = args.path
      print(self.path)
      self.path_model = self.path+'/model/'
      print(self.path_model)
      if not os.path.exists(self.path_model):
        os.makedirs(self.path_model)
      self.path_plot = self.path+'/plot/'
      if not os.path.exists(self.path_plot):
        os.makedirs(self.path_plot)
      #CREATE COVARIANCE MATRIX TO GET ACTION
      self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
      self.cov_mat = torch.diag(self.cov_var)
     

    def __init__(self, env, args):
      self.env = env
      self._init_hyperparameters(env,args)
      # Extract environment information
      self.actor = FFNetwork(self.obs_dim,self.act_dim)
      self.critic = FFNetwork(self.obs_dim,1)
      self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
      self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)
      self.episode_idx = []
      self.episode_R = []
      if args.seed != None:
        self.seed = args.seed
        # Check if our seed is valid first
        assert(type(self.seed) == int)
        # Set the seed 
        torch.manual_seed(self.seed)
        print(f"Successfully set seed to {self.seed}")
    #===========================================================================
    def get_action(self, states):
      mean = self.actor(states)
      dist = MultivariateNormal(mean, self.cov_mat)
      action = dist.sample()
      log_prob = dist.log_prob(action)
      return action.detach().numpy(), log_prob.detach()
    #===========================================================================
    def computer_reward_to_go(self, episode_rewards):
      ep_reward_to_go = []
      for ep_R in reversed(episode_rewards):
        discounted_R = 0
        for R in reversed(ep_R):
          discounted_R = R + discounted_R * self.gamma
          ep_reward_to_go.insert(0,discounted_R)
      R = ep_reward_to_go
      ep_reward_to_go = torch.tensor(ep_reward_to_go, dtype=torch.float)
      return ep_reward_to_go, R
    #===========================================================================
    def evaluate_value(self, episode_states, episode_actions):
      # CALCULATE STATE VALUE FUNCTIN V(S)
      V = self.critic(episode_states).squeeze()  
      # CALCULATE LOG PROBABILITY OF ACTIONS 
      mean = self.actor(episode_states)
      dist = MultivariateNormal(mean, self.cov_mat)
      log_prob = dist.log_prob(episode_actions)
      return V, log_prob
    #===========================================================================
    def rollout(self):
      # Batch data
      episode_states = []           # batch observations
      episode_actions = []         # batch actions
      episode_log_probs = []       # log probs of each action
      episode_rewards = []         # batch rewards
      episode_reward_to_go = []    # batch rewards-to-go
      episode_lengths = []         # episodic lengths in batch
      t = 0
      av_R = 0
      while t < self.steps_per_batch:
        ep_R = []
        state = self.env.reset()
        act_mat = np.zeros(self.act_dim)
        done = False
        for ep_t in range(self.steps_per_episode):
          t+=1
          episode_states.append(state)
          action, log_prob = self.get_action(state)
          next_state, reward, done, _ = self.env.step(action)
          av_R+=reward
          ep_R.append(reward)
          # print(action)
          episode_actions.append(action)
          episode_log_probs.append(log_prob)
          if done:
            break
        episode_lengths.append(ep_t+1)
        episode_rewards.append(ep_R)
      episode_states = torch.tensor(episode_states, dtype=torch.float)
      episode_actions = torch.tensor(episode_actions, dtype=torch.float)
      episode_log_probs = torch.tensor(episode_log_probs, dtype=torch.float)
      episode_reward_to_go, R = self.computer_reward_to_go(episode_rewards)
      return episode_states, episode_actions, episode_log_probs, episode_reward_to_go, episode_lengths, R
      #===========================================================================

    def plot(self):
        plt.figure()
        plt.plot(range(len(self.episode_R)), self.episode_R)
        plt.xlabel('No of Episodes+{}'.format(self.save_freq))
        plt.ylabel('Rewards')
        plt.savefig(self.path_plot + 'data_plot.png', format='png')

#===========================================================================
#===========================================================================
#===========================================================================

    def learn(self):
        t = 0 # to check total_time_step
        i = 0 # to check total_iteration/ episodes
        while t<self.total_step:
          ep_s, ep_a, ep_log_p, ep_rtg, ep_len, R = self.rollout()
          av_R = sum(R)/len(R)
          # print("Avg R:",av_R)
          # print('episode:', i, 'reward:', ep_rtg.mean())
          t += np.sum(ep_len)
          i += 1
          V, _ = self.evaluate_value(ep_s, ep_a)
          #calculate advantage
          Advantage_k = ep_rtg - V.detach() 
          #normalize advantage
          Advantage_k = (Advantage_k - Advantage_k.mean()) / (Advantage_k.std()+1e-10) 
          
          #=======RUN epoches per iteration =================
          for _ in range(self.epoch_per_iteration):
          
          #==============ACTOR===============  
            #=======calculate ratio 𝛑𝛳(a|s)/𝛑𝛳k(a|s)==========
            # 𝛑𝛳(a|s) = current log probability = curr_log_prob = log(𝛑𝛳(a|s))
            # 𝛑𝛳k(a|s) = episode log probability = ep_log_p = log(𝛑𝛳k(a|s))
            # log(𝛑𝛳(a|s)/𝛑𝛳k(a|s)) = log(𝛑𝛳(a|s)) - log(𝛑𝛳k(a|s))
            V_critic, curr_log_prob = self.evaluate_value(ep_s, ep_a)
            surrogate_ratios = torch.exp(curr_log_prob - ep_log_p) 
            surr_loss1 = surrogate_ratios * Advantage_k
            surr_loss2 = torch.clamp(surrogate_ratios, 1-self.clip, 1+self.clip)*Advantage_k
            actor_loss = (-torch.min(surr_loss1,surr_loss2)).mean()
            #==========Optimize Gradient=============
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
          #==============ACTOR===============

          #==============CRITIC==============
            critic_loss = nn.MSELoss()(V_critic, ep_rtg)
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()
          #==============CRITIC==============
          if i % self.save_freq == 0:
            print("Episode:",i, "Reward:", av_R)
            torch.save(self.actor.state_dict(),self.path_model+'actor')
            torch.save(self.critic.state_dict(),self.path_model+'critic')
            self.episode_R.append(av_R)
            self.plot()


       



    
