
# next is to add accel and see the difference
# add stiffness too
import numpy as np
from scipy import signal, stats
from matplotlib import pyplot as plt
from all_functions import *
import pickle
import os
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)

experiment_ID = "experiment_2"
errors_all_cyc_A_A = np.load("./results/{}/errors_all_cyc_A_A.npy".format(experiment_ID))
errors_all_cyc_A_B = np.load("./results/{}/errors_all_cyc_A_B.npy".format(experiment_ID))
errors_all_cyc_B_B = np.load("./results/{}/errors_all_cyc_B_B.npy".format(experiment_ID))
errors_all_p2p_A_A = np.load("./results/{}/errors_all_p2p_A_A.npy".format(experiment_ID))
errors_all_p2p_A_B = np.load("./results/{}/errors_all_p2p_A_B.npy".format(experiment_ID))
errors_all_p2p_B_B = np.load("./results/{}/errors_all_p2p_B_B.npy".format(experiment_ID))

## cyc
print("errors_mean: ",errors_all_cyc_A_A.mean(2))
print("errors_std: ",errors_all_cyc_A_A.std(2))
print("errors_mean: ",errors_all_cyc_A_B.mean(2))
print("errors_std: ",errors_all_cyc_A_B.std(2))
print("errors_mean: ",errors_all_cyc_B_B.mean(2))
print("errors_std: ",errors_all_cyc_B_B.std(2))
[f_ow, p_val_avg] = stats.f_oneway(errors_all_cyc_A_A.mean(0)[0],errors_all_cyc_A_B.mean(0)[0])
print("p-value (babbling/average/A_A vs A_B): ", p_val_avg)
[f_ow, p_val_avg] = stats.f_oneway(errors_all_cyc_A_A.mean(0)[1],errors_all_cyc_A_B.mean(0)[1])
print("p-value (refined/average/A_A vs A_B): ", p_val_avg)
[f_ow, p_val_avg] = stats.f_oneway(errors_all_cyc_A_A.mean(0)[1],errors_all_cyc_B_B.mean(0)[1])
print("p-value (refined/average/A_A vs B_B): ", p_val_avg)
#import pdb; pdb.set_trace()
y_lim=[0, .7]
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 5))
p0 = axes[0][0].boxplot(
	[errors_all_cyc_A_A.mean(0)[0], errors_all_cyc_A_B.mean(0)[0], errors_all_cyc_B_B.mean(0)[0]],
	notch=True,
	patch_artist=True)
axes[0][0].set_title(r'$(q_0+q_1)/2$',fontsize=12)
axes[0][0].set_ylim(y_lim)
#axes[0].set_xlabel('stiffness')
axes[0][0].set_xticklabels(["A_A", "A_B", "B_B"], rotation=45, fontsize=8)
axes[0][0].set_ylabel('RMSE')
p1 = axes[0][1].boxplot(
	[errors_all_cyc_A_A[0,0,:], errors_all_cyc_A_B[0,0,:], errors_all_cyc_B_B[0,0,:]],
	notch=True,
	patch_artist=True)
axes[0][1].set_title('$q_0$', fontsize=12)
axes[0][1].set_ylim(y_lim)
axes[0][1].set_yticklabels([])
#axes[1].set_xlabel('stiffness')
axes[0][1].set_xticklabels(["A_A", "A_B", "B_B"], rotation=45, fontsize=8)
p2 = axes[0][2].boxplot(
	[errors_all_cyc_A_A[1,0,:], errors_all_cyc_A_B[1,0,:], errors_all_cyc_B_B[1,0,:]],
	notch=True,
	patch_artist=True)
axes[0][2].set_title('$q_1$', fontsize=12)
axes[0][2].set_ylim(y_lim)
axes[0][2].set_yticklabels([])
#axes[2].set_xlabel('stiffness')
axes[0][2].set_xticklabels(["A_A", "A_B", "B_B"], rotation=45, fontsize=8)

p3 = axes[1][0].boxplot(
	[errors_all_cyc_A_A.mean(0)[-1], errors_all_cyc_A_B.mean(0)[-1], errors_all_cyc_B_B.mean(0)[-1]],
	notch=True,
	patch_artist=True)
#axes[1][0].set_title(r'$(q_0+q_1)/2$',fontsize=12)
axes[1][0].set_ylim(y_lim)
#axes[0].set_xlabel('stiffness')
axes[1][0].set_xticklabels(["A_A", "A_B", "B_B"], rotation=45, fontsize=8)
axes[1][0].set_ylabel('RMSE')
p4 = axes[1][1].boxplot(
	[errors_all_cyc_A_A[0,-1,:], errors_all_cyc_A_B[0,-1,:], errors_all_cyc_B_B[0,-1,:]],
	notch=True,
	patch_artist=True)
#axes[1][1].set_title('$q_0$', fontsize=12)
axes[1][1].set_ylim(y_lim)
axes[1][1].set_yticklabels([])
#axes[1].set_xlabel('stiffness')
axes[1][1].set_xticklabels(["A_A","A_B", "B_B"], rotation=45, fontsize=8)
p5 = axes[1][2].boxplot(
	[errors_all_cyc_A_A[1,-1,:], errors_all_cyc_A_B[1,-1,:], errors_all_cyc_B_B[1,-1,:]],
	notch=True,
	patch_artist=True)
#axes[1][2].set_title('$q_1$', fontsize=12)
axes[1][2].set_ylim(y_lim)
axes[1][2].set_yticklabels([])
#axes[2].set_xlabel('stiffness')
axes[1][2].set_xticklabels(["A_A","A_B","B_B"], rotation=45, fontsize=8)

for i_row in range(2):
	for j_col in range(3):
		axes[i_row][j_col].grid(True)
plt.show()
## p2p
print("errors_mean: ",errors_all_p2p_A_A.mean(2))
print("errors_std: ",errors_all_p2p_A_A.std(2))
print("errors_mean: ",errors_all_p2p_A_B.mean(2))
print("errors_std: ",errors_all_p2p_A_B.std(2))
print("errors_mean: ",errors_all_p2p_B_B.mean(2))
print("errors_std: ",errors_all_p2p_B_B.std(2))
[f_ow, p_val_avg] = stats.f_oneway(errors_all_p2p_A_A.mean(0)[0],errors_all_p2p_A_B.mean(0)[0])
print("p-value (babbling/average/A_A vs A_B): ", p_val_avg)
[f_ow, p_val_avg] = stats.f_oneway(errors_all_p2p_A_A.mean(0)[1],errors_all_p2p_A_B.mean(0)[1])
print("p-value (refined/average/A_A vs A_B): ", p_val_avg)
[f_ow, p_val_avg] = stats.f_oneway(errors_all_p2p_A_A.mean(0)[1],errors_all_p2p_B_B.mean(0)[1])
print("p-value (refined/average/A_A vs B_B): ", p_val_avg)
#import pdb; pdb.set_trace()

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 5))
p0 = axes[0][0].boxplot(
	[errors_all_p2p_A_A.mean(0)[0], errors_all_p2p_A_B.mean(0)[0], errors_all_p2p_B_B.mean(0)[0]],
	notch=True,
	patch_artist=True)
axes[0][0].set_title(r'$(q_0+q_1)/2$',fontsize=12)
axes[0][0].set_ylim(y_lim)
#axes[0].set_xlabel('stiffness')
axes[0][0].set_xticklabels(["A_A", "A_B", "B_B"], rotation=45, fontsize=8)
axes[0][0].set_ylabel('RMSE')
p1 = axes[0][1].boxplot(
	[errors_all_p2p_A_A[0,0,:], errors_all_p2p_A_B[0,0,:], errors_all_p2p_B_B[0,0,:]],
	notch=True,
	patch_artist=True)
axes[0][1].set_title('$q_0$', fontsize=12)
axes[0][1].set_ylim(y_lim)
axes[0][1].set_yticklabels([])
#axes[1].set_xlabel('stiffness')
axes[0][1].set_xticklabels(["A_A", "A_B", "B_B"], rotation=45, fontsize=8)
p2 = axes[0][2].boxplot(
	[errors_all_p2p_A_A[1,0,:], errors_all_p2p_A_B[1,0,:], errors_all_p2p_B_B[1,0,:]],
	notch=True,
	patch_artist=True)
axes[0][2].set_title('$q_1$', fontsize=12)
axes[0][2].set_ylim(y_lim)
axes[0][2].set_yticklabels([])
#axes[2].set_xlabel('stiffness')
axes[0][2].set_xticklabels(["A_A", "A_B", "B_B"], rotation=45, fontsize=8)

p3 = axes[1][0].boxplot(
	[errors_all_p2p_A_A.mean(0)[-1], errors_all_p2p_A_B.mean(0)[-1], errors_all_p2p_B_B.mean(0)[-1]],
	notch=True,
	patch_artist=True)
#axes[1][0].set_title(r'$(q_0+q_1)/2$',fontsize=12)
axes[1][0].set_ylim(y_lim)
#axes[0].set_xlabel('stiffness')
axes[1][0].set_xticklabels(["A_A", "A_B", "B_B"], rotation=45, fontsize=8)
axes[1][0].set_ylabel('RMSE')
p4 = axes[1][1].boxplot(
	[errors_all_p2p_A_A[0,-1,:], errors_all_p2p_A_B[0,-1,:], errors_all_p2p_B_B[0,-1,:]],
	notch=True,
	patch_artist=True)
#axes[1][1].set_title('$q_0$', fontsize=12)
axes[1][1].set_ylim(y_lim)
axes[1][1].set_yticklabels([])
#axes[1].set_xlabel('stiffness')
axes[1][1].set_xticklabels(["A_A","A_B", "B_B"], rotation=45, fontsize=8)
p5 = axes[1][2].boxplot(
	[errors_all_p2p_A_A[1,-1,:], errors_all_p2p_A_B[1,-1,:], errors_all_p2p_B_B[1,-1,:]],
	notch=True,
	patch_artist=True)
#axes[1][2].set_title('$q_1$', fontsize=12)
axes[1][2].set_ylim(y_lim)
axes[1][2].set_yticklabels([])
#axes[2].set_xlabel('stiffness')
axes[1][2].set_xticklabels(["A_A","A_B","B_B"], rotation=45, fontsize=8)

for i_row in range(2):
	for j_col in range(3):
		axes[i_row][j_col].grid(True)
plt.show()

#import pdb; pdb.set_trace()
