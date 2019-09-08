
# next is to add accel and see the difference
# add stiffness too
import numpy as np
from scipy import signal, stats
from matplotlib import pyplot as plt
from all_functions import *
import pickle
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

experiment_ID="experiment_1B"

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9, 3))
N = 9
stiffness_values = ["0", "500", "1K", "2K", "4K", "7K", "10K", "15K", "20K"]
x = np.arange(N)    # the x locations for the groups

positions_cyclical = 2*np.arange(N)-0.25
positions_p2p = 2*np.arange(N)+0.25
y_lim=[0, .95]
y_lim_p0=[.05, .75]

## cyclical
file_address = "./results/{}/errors_all_cyclical.npy".format(experiment_ID)
errors_all = np.load(file_address)
errors_all_mean=errors_all.mean(1)
errors_all_std = errors_all.std(1)
print("errors_mean: ",errors_all_mean)
print("errors_std: ",errors_all_std)
p0_0 = axes[0].boxplot(
	errors_all.mean(0).reshape(1,-1,1,N).squeeze(),
	positions=positions_cyclical,
	notch=True,
	patch_artist=True)

p0_1 = axes[1].boxplot(
	errors_all[0,:,:,:].reshape(1,-1,1,N).squeeze(),
	positions=positions_cyclical,
	notch=True,
	patch_artist=True)

p0_2 = axes[2].boxplot(
	errors_all[1,:,:,:].reshape(1,-1,1,N).squeeze(),
	positions=positions_cyclical,
	notch=True,
	patch_artist=True)
# p-value
[f_ow, p_val] = stats.f_oneway(errors_all.mean(0)[:,0,0],errors_all.mean(0)[:,0,3])
print("p-value: ", p_val)

## p2p
file_address = "./results/{}/errors_all_p2p.npy".format(experiment_ID)
errors_all = np.load(file_address)
errors_all_mean=errors_all.mean(1)
errors_all_std = errors_all.std(1)
print("errors_mean: ",errors_all_mean)
print("errors_std: ",errors_all_std)
p1_0 = axes[0].boxplot(
	errors_all.mean(0).reshape(1,-1,1,N).squeeze(),
	positions=positions_p2p,
	notch=True,
	patch_artist=True)
axes[0].set_title(r'average across both joints',fontsize=10)
axes[0].set_ylim(y_lim_p0)
axes[0].set_xlabel('stiffness (N/m)')
axes[0].set_xticklabels(stiffness_values, rotation=45, fontsize=8)
axes[0].set_ylabel('RMSE')

p1_1 = axes[1].boxplot(
	errors_all[0,:,:,:].reshape(1,-1,1,N).squeeze(),
	positions=positions_p2p,
	notch=True,
	patch_artist=True)
axes[1].set_title('proximal joint ($q_0$)', fontsize=10)
axes[1].set_ylim(y_lim)
axes[1].set_yticklabels([])
axes[1].set_xlabel('stiffness (N/m)')
axes[1].set_xticklabels(stiffness_values, rotation=45, fontsize=8)

p1_2 = axes[2].boxplot(
	errors_all[1,:,:,:].reshape(1,-1,1,N).squeeze(),
	positions=positions_p2p,
	notch=True,
	patch_artist=True)
axes[2].set_title('distal joint ($q_1$)', fontsize=10)
axes[2].set_ylim(y_lim)
axes[2].set_yticklabels([])
axes[2].set_xlabel('stiffness (N/m)')
axes[2].set_xticklabels(stiffness_values, rotation=45, fontsize=8)

# changing the box colors
for bplot in (p1_0, p1_1, p1_2):
    for patch in bplot['boxes']:
        patch.set_facecolor('lightskyblue')

axes[2].legend([p0_2["boxes"][0], p1_2["boxes"][0]], ['cyclical','point-to-point'], loc='upper right', fontsize='small')#bbox_to_anchor=(1.6, 1.02)
fig.subplots_adjust(left=.06, right=.95, bottom=.17)
fig.savefig('./results/{}/error_vs_stiffness.png'.format(experiment_ID))
# p-value
[f_ow, p_val] = stats.f_oneway(errors_all.mean(0)[:,0,0],errors_all.mean(0)[:,0,3])
print("p-value: ", p_val)
##
plt.show()
#import pdb; pdb.set_trace()

