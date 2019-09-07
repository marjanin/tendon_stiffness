import gym
import numpy as np

from stable_baselines.common.policies import MlpPolicy as common_MlpPolicy
from stable_baselines.ddpg.policies import MlpPolicy as DDPG_MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import PPO1, PPO2, DDPG
import multiprocessing as mp
import tensorflow as tf
from gym.utils import seeding
import random

def ppo1_nmileg_pool(stiffness_value):
	RL_method = "PPO1"
	#total_MC_runs = 50
	experiment_ID = "experiment_4_pool_with_MC_C/"
	save_name_extension = RL_method
	total_timesteps = 500000
	stiffness_value_str = "stiffness_{}".format(stiffness_value)
	current_mc_run_num = 15 # starts from 0
	for mc_cntr in range(current_mc_run_num, current_mc_run_num+1):
		log_dir = "./logs/{}/MC_{}/{}/{}/".format(experiment_ID, mc_cntr, RL_method, stiffness_value_str)
		# defining the environments
		env = gym.make('TSNMILeg{}-v1'.format(stiffness_value))
		#env = gym.wrappers.Monitor(env, "./tmp/gym-results", video_callable=False, force=True)
		## setting the Monitor
		env = gym.wrappers.Monitor(env, log_dir+"Monitor/", video_callable=False, force=True, uid="Monitor_info")
		# defining the initial model
		if RL_method == "PPO1":
			model = PPO1(common_MlpPolicy, env, verbose=1, tensorboard_log=log_dir)
		elif RL_method == "PPO2":
			env = DummyVecEnv([lambda: env])
			model = PPO2(common_MlpPolicy, env, verbose=1, tensorboard_log=log_dir)
		elif RL_method == "DDPG":
			env = DummyVecEnv([lambda: env])
			n_actions = env.action_space.shape[-1]
			param_noise = None
			action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5)* 5 * np.ones(n_actions))
			model = DDPG(DDPG_MlpPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise, tensorboard_log=log_dir)
		else:
			raise ValueError("Invalid RL mode")
		# setting the environment on the model
		#model.set_env(env)
		# setting the random seed for some of the random instances
		random_seed = mc_cntr
		random.seed(random_seed)
		env.seed(random_seed)
		env.action_space.seed(random_seed)
		np.random.seed(random_seed)
		tf.random.set_random_seed(random_seed)
		# training the model
		# training the model
		model.learn(total_timesteps=total_timesteps)
		# saving the trained model
		model.save(log_dir+"/model")
	return None

pool = mp.Pool(mp.cpu_count())

stiffness_versions = 9
pool.map_async(ppo1_nmileg_pool, [row for row in range(stiffness_versions)])
pool.close()
pool.join()
#import pdb; pdb.set_trace()
