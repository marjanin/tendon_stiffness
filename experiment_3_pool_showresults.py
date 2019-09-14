
# next is to add accel and see the difference
# add stiffness too
import tensorflow as tf
import numpy as np
from scipy import signal, stats
from matplotlib import pyplot as plt
from all_functions import *
import pickle
from warnings import simplefilter

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['mathtext.fontset'] = 'cm'

plt.rcParams['svg.fonttype'] = 'none'
def calculate_mean_std(data, method='mean'):
	stiffness_versions = data.shape[0]
	output = np.zeros(stiffness_versions)
	for stiffness_version_A in range(stiffness_versions):
		if method == 'mean':
			output[stiffness_version_A] = data[stiffness_version_A].mean(0)
		elif method == 'std':
			output[stiffness_version_A] = data[stiffness_version_A].std(0)
		else:
			raise NameError('invalid method: please use mean or std')
	return output



experiment_ID = "experiment_3_pool_G"

if experiment_ID == "experiment_3_pool_B":
	reward_thresh = 5 #3:7 3B:5
	mc_run_number = 50
elif experiment_ID == "experiment_3_pool_D":
	reward_thresh = 5 #3:7 3B:5
	mc_run_number = 100
elif experiment_ID == "experiment_3_pool_C":
	reward_thresh = 6 #3:7 3B:5
	mc_run_number = 50
elif experiment_ID == "experiment_3_pool_E":
	reward_thresh = 4 #3:7 3B:5
	mc_run_number = 20
elif experiment_ID == "experiment_3_pool_F":
	reward_thresh = 3 #3:7 3B:5
	mc_run_number = 20
elif experiment_ID == "experiment_3_pool_G":
	reward_thresh = 3 #3:7 3B:5
	mc_run_number = 100
else:
	reward_thresh = 7 #3:7 3B:5
	mc_run_number = 50

stiffness_versions = 9

exploration_run_numbers_all = np.zeros([mc_run_number, stiffness_versions])
rewards_all = np.zeros([mc_run_number, stiffness_versions])
energies_all = np.zeros([mc_run_number, stiffness_versions])


for stiffness_version_A in range(stiffness_versions):
	exploration_run_numbers_all [:, stiffness_version_A] = np.load("./results/{}/exploration_run_numbers_S{}.npy".format(experiment_ID, stiffness_version_A))
	rewards_all [:, stiffness_version_A] = np.load("./results/{}/rewards_S{}.npy".format(experiment_ID, stiffness_version_A))
	energies_all [:, stiffness_version_A] = np.load("./results/{}/energies_S{}.npy".format(experiment_ID, stiffness_version_A))

#y_lim=[0, .7]


number_of_successful_attempt = np.array(sum(rewards_all>=reward_thresh))
successful_rewards = []
successful_energies = []
#number_of_successful_attempt = np.where(rewards_all>=7)
for stiffness_version_A in range(stiffness_versions):
	successful_rewards.append(rewards_all[rewards_all[:, stiffness_version_A]>=reward_thresh, stiffness_version_A])
	successful_energies.append(energies_all[rewards_all[:, stiffness_version_A]>=reward_thresh, stiffness_version_A])
successful_rewards = np.array(successful_rewards)
successful_energies = np.array(successful_energies)

successful_rewards_means = calculate_mean_std(successful_rewards, method='mean')
successful_rewards_stds = calculate_mean_std(successful_rewards, method='std')
successful_energies_means = calculate_mean_std(successful_energies, method='mean')
successful_energies_stds = calculate_mean_std(successful_energies, method='std')

[f_ow, p_val_avg] = stats.f_oneway(successful_energies[0],successful_energies[4])
print("p-value (S = 0 & S = 4K): ", p_val_avg)
[f_ow, p_val_avg] = stats.f_oneway(successful_energies[6],successful_energies[4])
print("p-value (S = 10K & S = 4K): ", p_val_avg)

nrows = 1
ncols = 3
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(9, 3))
#import pdb; pdb.set_trace()
p0 = axes[0].bar(range(stiffness_versions),number_of_successful_attempt/mc_run_number, alpha=.5)
p1 = axes[1].bar(range(stiffness_versions),successful_rewards_means,yerr=successful_rewards_stds, alpha=.5)
p2 = axes[2].bar(range(stiffness_versions),successful_energies_means,yerr=successful_energies_stds, alpha=.5)
#axes[1].set_xlabel('stiffness')
#axes[1].set_xticklabels(stiffness_values, rotation=45, fontsize=8)
stiffness_values = ["0", "500", "1K", "2K", "4K", "7K", "10K", "15K", "20K"]
#titles = []
xlabels = ['Stiffness']*3
ylabels = ['Success rate (%)', 'Reward', 'Energy']

for ii in range(ncols):
	plt.sca(axes[ii])
	plt.xlabel(xlabels[ii], fontsize=9)
	plt.xticks(range(stiffness_versions), stiffness_values, rotation=45, fontsize=8)
	plt.ylabel(ylabels[ii], fontsize=9)
	plt.yticks(rotation=45, fontsize=8)
fig.subplots_adjust(top=.9, bottom=.2, left=.06, right=.95, wspace=.30)
fig.savefig('./results/{}/exp3_learn2walk_results.pdf'.format(experiment_ID))
fig.savefig('./results/figures/exp3_learn2walk_results.pdf'.format(experiment_ID))
# p0 = axes[1].boxplot(
# 	[errors_all_cyc_A_A.mean(0)[0], errors_all_cyc_A_B.mean(0)[0], errors_all_cyc_B_B.mean(0)[0]],
# 	notch=True,
# 	patch_artist=True)
plt.show()

#import pdb; pdb.set_trace()
