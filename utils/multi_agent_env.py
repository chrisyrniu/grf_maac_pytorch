import gym
import gfootball.env as grf_env
import numpy as np

class MultiAgentEnv(gym.Env):

	def __init__(self, scenario, num_controlled_lagents, num_controlled_ragents, reward_type, render):
		self.env = grf_env.create_environment(
			env_name=scenario,
			stacked=False,
			representation='multiagent',
			rewards=reward_type,
			write_goal_dumps=False,
			write_full_episode_dumps=False,
			render=render,
			dump_frequency=0,
			logdir='/tmp/maddpg_test',
			extra_players=None,
			number_of_left_players_agent_controls=num_controlled_lagents,
			number_of_right_players_agent_controls=num_controlled_ragents,
			channel_dimensions=(3, 3)
			)
		self.num_controlled_lagents = num_controlled_lagents
		self.num_controlled_ragents = num_controlled_ragents
		self.num_controlled_agents = num_controlled_lagents + num_controlled_ragents
		self.num_lagents = self.env.num_lteam_players
		self.num_ragents = self.env.num_rteam_players
		if self.num_controlled_agents > 1:
			action_space = gym.spaces.Discrete(self.env.action_space.nvec[1])
		else:
			action_space = self.env.action_space
		observation_space = gym.spaces.Box(
	        low=self.env.observation_space.low[0],
	        high=self.env.observation_space.high[0],
			dtype=self.env.observation_space.dtype)
		self.action_space = [action_space for _ in range(self.num_controlled_agents)]
		self.observation_space = [observation_space for _ in range(self.num_controlled_agents)]

	def seed(self, seed=None):
		if seed is None:
			np.random.seed(1)
		else:
			np.random.seed(seed)

	# def extract_obs(self, original_obs):
	# 	lteam_pos = original_obs[0 : 2*self.num_lagents]
	# 	rteam_pos = original_obs[4*self.num_lagents : 4*self.num_lagents+2*self.num_ragents]
	# 	ball_pos = original_obs[4*self.num_lagents+4*self.num_ragents : 4*self.num_lagents+4*self.num_ragents+3]
	# 	ball_side = original_obs[4*self.num_lagents+4*self.num_ragents+6 : 4*self.num_lagents+4*self.num_ragents+9]
	# 	game_mode = original_obs[4*self.num_lagents+4*self.num_ragents+9 : 4*self.num_lagents+4*self.num_ragents+16]
	# 	return np.concatenate((lteam_pos, rteam_pos, ball_pos, ball_side, game_mode))	

	def reset(self):
		original_obs = self.env.reset()
		obs = []
		for x in range(self.num_controlled_agents):
			if self.num_controlled_agents > 1:
				obs.append(original_obs[x])
			else:
				obs = original_obs
		return obs

	def step(self, actions):
		o, r, d, i = self.env.step(actions)
		next_obs = [o[x] for x in range(len(o))]
		rewards = r
		dones = [d for x in range(len(o))]
		infos = i

		return next_obs, rewards, dones, infos

