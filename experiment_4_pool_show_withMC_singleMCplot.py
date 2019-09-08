import tensorflow as tf
import json
import numpy as np
from scipy import signal, stats
from matplotlib import pyplot as plt
import pickle
from warnings import simplefilter
from stable_baselines.results_plotter import load_results, ts2xy

stiffness_versions = 9
RL_method = "PPO1"
total_MC_runs = 50 # starts from 1
experiment_ID = "experiment_4_pool_with_MC_C"
total_timesteps = 500000
episode_timesteps = 1000
total_episodes = int(total_timesteps/episode_timesteps)

episode_rewards_all = np.zeros([total_MC_runs, stiffness_versions, total_episodes])

for stiffness_value in range(stiffness_versions):
	stiffness_value_str = "stiffness_{}".format(stiffness_value)
	for mc_cntr in range(total_MC_runs-1,total_MC_runs):
		log_dir = "./logs/{}/MC_{}/{}/{}/".format(experiment_ID, mc_cntr, RL_method, stiffness_value_str)
		jsonFile = open(log_dir+"monitor/openaigym.episode_batch.{}.Monitor_info.stats.json".format(0))
		jsonString = jsonFile.read()
		jsonData = json.loads(jsonString)
		print("stiffness_value: ", stiffness_value, "mc_cntr: ", mc_cntr)
		episode_rewards_all[mc_cntr, stiffness_value, :] = np.array(jsonData['episode_rewards'])
episode_rewards_average = episode_rewards_all.mean(0)

for stiffness_value in range(stiffness_versions):
	plt.plot(episode_rewards_average[stiffness_value,:])
plt.legend(['0','500', '1K', '2K', '4K', '7K', '10K', '15K', '20K'])
plt.show()

#import pdb; pdb.set_trace()
