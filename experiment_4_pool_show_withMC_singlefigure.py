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
matplotlib.rcParams['mathtext.fontset'] = 'cm'

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

#fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4.5, 3))
plt.figure(figsize=(9,3))
plt.subplot(121)
stiffness_values_full = ["0", "500", "1k", "2k", "4k", "7k", "10k", "15k", "20k"]
stiffness_values_legend_full = ["K: 0", "K: 500", "K: 1k", "K: 2k", "K: 4k", "K: 7k", "K: 10k", "K: 15k", "K: 20k"]

if extended_view:
	stiffness_values = stiffness_values_full
	stiffness_values_legend = stiffness_values_legend_full
else:
	stiffness_values = ["0", "1k", "4k", "10k", "20k"]
	stiffness_values_legend = ["K: 0", "K: 1k", "K: 4k", "K: 10k", "K: 20k"]
## Figure 1
for stiffness_value in select_stiffness:
	x0=range(total_episodes)
	y0=episode_displacement_average[stiffness_value,:]
	std0 = episode_displacement_std[stiffness_value,:]
	plt.plot(x0, y0, color=colorsys.hsv_to_rgb((8.75-stiffness_value)/14,1,.75), alpha=.75)
	plt.fill_between(x0, y0-std0/2, y0+std0/2,
		color=colorsys.hsv_to_rgb((8.75-stiffness_value)/14,1,.75), alpha=0.20)
plt.legend(stiffness_values_legend, fontsize='x-small',loc='lower right')
plt.xlabel('Episode #', fontsize=8)
plt.ylabel('Displacement (m)', fontsize=8)
plt.xticks(np.arange(0,501,50),rotation=45, fontsize=8)
plt.yticks(rotation=45, fontsize=8)
plt.title('a) Learning curves: reward vs. episode plots', fontsize=8)
#plt.grid()


for stiffness_value in range(stiffness_versions):	
	final_displacement[:,stiffness_value] = episode_displacement_all[:,stiffness_value,-1]
	for mc_cntr in range(total_MC_runs):
		pass_displacement_point[mc_cntr, stiffness_value] = np.min(np.where(episode_displacement_all[mc_cntr, stiffness_value,:]>=displacement_point))

#fig.subplots_adjust(top=.98, bottom=.15, left=.13, right=.95, wspace=.33)
#fig.savefig('./results/{}/PPO_results_1.png'.format(experiment_ID))
## Figure 2
plt.subplot(143)
x1=range(stiffness_versions)
y1 = pass_displacement_point.mean(0)
std1 = pass_displacement_point.std(0)
plt.plot(x1, y1, '--',color='black',alpha=.1)
#plt.fill_between(x1, y1-std1/2, y1+std1/2, alpha=0.25, edgecolor='C9', facecolor='C9')
plt.errorbar(x1, y1,yerr=std1/2,color='black',alpha=.2,animated=True)
for stiffness_value in range(stiffness_versions):
	plt.plot(x1[stiffness_value], y1[stiffness_value], 'o',alpha=1, color=colorsys.hsv_to_rgb((8.75-stiffness_value)/14,1,.75))
plt.xlabel('Stiffness (N/M)', fontsize=8)
plt.ylabel('Episode #', fontsize=8)
plt.xticks(range(stiffness_versions), stiffness_values_full, rotation=45, fontsize=8)
plt.yticks(rotation=45, fontsize=8)
plt.title('b) Passing threshold episode', fontsize=8)
#plt.grid()
## Figure 3
plt.subplot(144)
x2=range(stiffness_versions)
y2 = final_displacement.mean(0)
std2 = final_displacement.std(0)
plt.plot(x2, y2, '--',color='black',alpha=.1)
#plt.fill_between(x2, y2-std2/2, y2+std2/2, alpha=0.25, edgecolor='C9', facecolor='C9')
plt.errorbar(x2, y2,yerr=std2/2,color='black',alpha=.2,animated=True)
for stiffness_value in range(stiffness_versions):
	plt.plot(x2[stiffness_value], y2[stiffness_value], 'o',alpha=.9, color=colorsys.hsv_to_rgb((8.75-stiffness_value)/14,1,.75))
plt.xlabel('Stiffness (N/M)', fontsize=8)
plt.ylabel('Displacement (m)', fontsize=8)
plt.xticks(range(stiffness_versions), stiffness_values_full,  rotation=45, fontsize=8)
plt.yticks(rotation=45, fontsize=8)
plt.title('c) Average final rewards', fontsize=8)
#plt.grid()
# for ii in range(ncols):
# 	plt.sca(axes[ii])
# 	plt.xlabel(xlabels[ii], fontsize=9)
# 	plt.xticks(x1, stiffness_values_full, rotation=45, fontsize=8)
# # 	plt.ylabel(ylabels[ii], fontsize=9)
# # 	plt.yticks(rotation=45, fontsize=8)
# fig.subplots_adjust(top=.95, bottom=.17, left=.11, right=.95, wspace=.33)
# fig.savefig('./results/{}/PPO_results_2.png'.format(experiment_ID))
# p0 = axes[1].boxplot(
# 	[errors_all_cyc_A_A.mean(0)[0], errors_all_cyc_A_B.mean(0)[0], errors_all_cyc_B_B.mean(0)[0]],
# 	notch=True,
# 	patch_artist=True)
os.makedirs("./results/{}".format(experiment_ID), exist_ok=True)
plt.subplots_adjust(left=0.06, bottom=0.16, right=0.95, top=0.92, wspace=0.32)
plt.savefig("./results/{}/exp4_PPO_results_combined.pdf".format(experiment_ID))
plt.savefig("./results/figures/exp4_PPO_results_combined.pdf".format(experiment_ID))
plt.show() 


#import pdb; pdb.set_trace()
