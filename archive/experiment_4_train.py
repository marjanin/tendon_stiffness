import gym
import numpy as np

from stable_baselines.common.policies import MlpPolicy as common_MlpPolicy
from stable_baselines.ddpg.policies import MlpPolicy as DDPG_MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import PPO1, PPO2, DDPG
#defining the variables
RL_method = "PPO1"
experiment_ID = "experiment_4_test"

save_name_extension = RL_method
total_timesteps = 1000
stiffness_versions = 9
for stiffness_value in range(stiffness_versions):
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

# ## running the trained model
# # remove to demonstrate saving and loading
# del model 
# # defining the environments
# su_env = gym.make('HalfCheetah_nssu-v3')
# su_env = DummyVecEnv([lambda: su_env])
# ru_env = gym.make('HalfCheetah_nsru-v3')
# ru_env = DummyVecEnv([lambda: ru_env])
# # loading the trained model
# if RL_method == "PPO2":
# 	model = PPO2.load("trainedmodel-HalfCheetah_nssuru_"+save_name_extension)
# elif RL_method == "DDPG":
# 	model = DDPG.load("trainedmodel-HalfCheetah_nssuru_"+save_name_extension)
# else:
# 	raise ValueError("Invalid RL mode")
# # setting the seocond environment
# model.set_env(ru_env)
# #model = DDPG.load("PPO2-HalfCheetah_nssu-v3_test2")
# obs = ru_env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = ru_env.step(action)
#     ru_env.render()

#import pdb; pdb.set_trace()
#tensorboard --logdir=/Users/alimarjaninejad/Documents/github/marjanin/gym_ali/log/
#http://Alis-MacBook-Pro.local:6006