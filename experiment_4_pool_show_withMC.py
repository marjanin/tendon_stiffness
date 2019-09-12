import json
import numpy as np
from scipy import signal, stats
from matplotlib import pyplot as plt
import colorsys
import pickle
from warnings import simplefilter
import os

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

extended_view = False
stiffness_versions = 9
if extended_view:
	select_stiffness = range(stiffness_versions)
else:
	select_stiffness = np.array([0, 2, 4, 6, 8])
RL_method = "PPO1"
total_MC_runs = 50 # starts from 1
experiment_ID = "experiment_4_pool_with_MC_C"
total_timesteps = 500000
episode_timesteps = 1000
total_episodes = int(total_timesteps/episode_timesteps)

episode_rewards_all = np.zeros([total_MC_runs, stiffness_versions, total_episodes])

for stiffness_value in range(stiffness_versions):
	stiffness_value_str = "stiffness_{}".format(stiffness_value)
	for mc_cntr in range(total_MC_runs):
		log_dir = "./logs/{}/MC_{}/{}/{}/".format(experiment_ID, mc_cntr, RL_method, stiffness_value_str)
		jsonFile = open(log_dir+"monitor/openaigym.episode_batch.{}.Monitor_info.stats.json".format(0))
		jsonString = jsonFile.read()
		jsonData = json.loads(jsonString)
		print("stiffness_value: ", stiffness_value, "mc_cntr: ", mc_cntr)
		episode_rewards_all[mc_cntr, stiffness_value, :] = np.array(jsonData['episode_rewards'])
reward_to_displacement_coeficient  = .01
episode_displacement_all = episode_rewards_all*reward_to_displacement_coeficient
episode_displacement_average = episode_displacement_all.mean(0)
episode_displacement_std = episode_displacement_all.std(0)
final_displacement = np.zeros([total_MC_runs, stiffness_versions])
pass_displacement_point = np.zeros([total_MC_runs ,stiffness_versions])
displacement_point = 9
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4.5, 3))

stiffness_values_full = ["0", "500", "1K", "2K", "4K", "7K", "10K", "15K", "20K"]
if extended_view:
	stiffness_values = stiffness_values_full
else:
	stiffness_values = ["0", "1K", "4K", "10K", "20K"]
## Figure 1
for stiffness_value in select_stiffness:
	x0=range(total_episodes)
	y0=episode_displacement_average[stiffness_value,:]
	std0 = episode_displacement_std[stiffness_value,:]
	plt.plot(x0, y0, color=colorsys.hsv_to_rgb((8-stiffness_value)/14,1,.75), alpha=.75)
	plt.fill_between(x0, y0-std0/2, y0+std0/2,
		color=colorsys.hsv_to_rgb((8-stiffness_value)/14,1,.75), alpha=0.20)
	

for stiffness_value in range(stiffness_versions):	
	final_displacement[:,stiffness_value] = episode_displacement_all[:,stiffness_value,-1]
	for mc_cntr in range(total_MC_runs):
		pass_displacement_point[mc_cntr, stiffness_value] = np.min(np.where(episode_displacement_all[mc_cntr, stiffness_value,:]>=displacement_point))

plt.legend(stiffness_values, fontsize='x-small',loc='lower right')
plt.xlabel('Episode #')
plt.ylabel('Displacement (m)')
plt.yticks(rotation=45, fontsize=8)

os.makedirs("./results/{}".format(experiment_ID), exist_ok=True)
fig.subplots_adjust(top=.98, bottom=.15, left=.13, right=.95, wspace=.33)
fig.savefig('./results/{}/PPO_results_1.png'.format(experiment_ID))
## Figure 2
ncols = 2
nrows = 1
fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(4.5, 3))
x1=range(stiffness_versions)
y1 = pass_displacement_point.mean(0)
std1 = pass_displacement_point.std(0)
axes[0].plot(x1, y1, '-')
axes[0].fill_between(x1, y1-std1/2, y1+std1/2, alpha=0.25, edgecolor='C9', facecolor='C9')

x2=range(stiffness_versions)
y2 = final_displacement.mean(0)
std2 = final_displacement.std(0)
axes[1].plot(x2, y2, '-')
axes[1].fill_between(x2, y2-std2/2, y2+std2/2, alpha=0.25, edgecolor='C9', facecolor='C9')
xlabels = ['Stiffness (N/M)', 'Stiffness (N/M)']
ylabels = ['Episode #', 'Displacement (m)']


for stiffness_value in range(stiffness_versions):
	axes[0].plot(x1[stiffness_value], y1[stiffness_value], 'o',alpha=.9, color=colorsys.hsv_to_rgb((8-stiffness_value)/14,1,.75))
	axes[1].plot(x2[stiffness_value], y2[stiffness_value], 'o',alpha=.9, color=colorsys.hsv_to_rgb((8-stiffness_value)/14,1,.75))


for ii in range(ncols):
	plt.sca(axes[ii])
	plt.xlabel(xlabels[ii], fontsize=9)
	plt.xticks(x1, stiffness_values_full, rotation=45, fontsize=8)
	plt.ylabel(ylabels[ii], fontsize=9)
	plt.yticks(rotation=45, fontsize=8)
fig.subplots_adjust(top=.95, bottom=.17, left=.11, right=.95, wspace=.33)
fig.savefig('./results/{}/PPO_results_2.png'.format(experiment_ID))
# p0 = axes[1].boxplot(
# 	[errors_all_cyc_A_A.mean(0)[0], errors_all_cyc_A_B.mean(0)[0], errors_all_cyc_B_B.mean(0)[0]],
# 	notch=True,
# 	patch_artist=True)
plt.show() 


#import pdb; pdb.set_trace()
