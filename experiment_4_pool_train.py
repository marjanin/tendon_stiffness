import gym
import numpy as np

from stable_baselines.common.policies import MlpPolicy as common_MlpPolicy
from stable_baselines.ddpg.policies import MlpPolicy as DDPG_MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import PPO1, PPO2, DDPG
import multiprocessing as mp
#defining the variables

def ppo1_nmileg_pool(stiffness_value):
	RL_method = "PPO1"
	experiment_ID = "experiment_4_pool_A/mc_1/"
	save_name_extension = RL_method
	total_timesteps = 500000
	stiffness_value_str = "stiffness_{}".format(stiffness_value)
	log_dir = "./logs/{}/{}/{}/".format(experiment_ID, RL_method, stiffness_value_str)
	# defining the environments
	env = gym.make('TSNMILeg{}-v1'.format(stiffness_value))
	#env = gym.wrappers.Monitor(env, "./tmp/gym-results", video_callable=False, force=True)
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
