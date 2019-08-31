
# next is to add accel and see the difference
# add stiffness too
import numpy as np
from scipy import signal, stats
from matplotlib import pyplot as plt
from all_functions import *
import pickle
from warnings import simplefilter

def exp2_learning_curves_cal_fcn(errors_all):
	average_curve_mean = errors_all.mean(0).mean(1)
	q0_curve_mean = errors_all[0].mean(1)
	q1_curve_mean = errors_all[1].mean(1)
	average_curve_std = errors_all.mean(0).std(1)
	q0_curve_std = errors_all[0].std(1)
	q1_curve_std = errors_all[1].std(1)
	return average_curve_mean, q0_curve_mean, q1_curve_mean, average_curve_std, q0_curve_std, q1_curve_std 

simplefilter(action='ignore', category=FutureWarning)
experiment_ID = "experiment_2"
number_of_refinements = 5
errors_all_cyc_A_A = np.load("./results/{}/errors_all_cyc_A_A.npy".format(experiment_ID))
errors_all_cyc_A_B = np.load("./results/{}/errors_all_cyc_A_B.npy".format(experiment_ID))
errors_all_cyc_B_B = np.load("./results/{}/errors_all_cyc_B_B.npy".format(experiment_ID))
errors_all_p2p_A_A = np.load("./results/{}/errors_all_p2p_A_A.npy".format(experiment_ID))
errors_all_p2p_A_B = np.load("./results/{}/errors_all_p2p_A_B.npy".format(experiment_ID))
errors_all_p2p_B_B = np.load("./results/{}/errors_all_p2p_B_B.npy".format(experiment_ID))

number_of_mods = 6
errors_all = np.zeros((number_of_mods,)+errors_all_cyc_A_A.shape)
average_curve_mean_all = np.zeros([number_of_mods,number_of_refinements+1])
q0_curve_mean_all = np.zeros([number_of_mods,number_of_refinements+1])
q1_curve_mean_all= np.zeros([number_of_mods,number_of_refinements+1])
average_curve_std_all = np.zeros([number_of_mods,number_of_refinements+1])
q0_curve_std_all = np.zeros([number_of_mods,number_of_refinements+1])
q1_curve_std_all= np.zeros([number_of_mods,number_of_refinements+1])
errors_all = \
np.array([errors_all_cyc_A_A,
errors_all_cyc_A_B,
errors_all_cyc_B_B,
errors_all_p2p_A_A,
errors_all_p2p_A_B,
errors_all_p2p_B_B])
print('hi')
for mod_iter in range(number_of_mods):
	[average_curve_mean_all[mod_iter,:], q0_curve_mean_all[mod_iter,:], q1_curve_mean_all[mod_iter,:], average_curve_std_all[mod_iter,:], q0_curve_std_all[mod_iter,:], q1_curve_std_all[mod_iter,:]] = \
	exp2_learning_curves_cal_fcn(errors_all=errors_all[mod_iter,:])
print('bye')
## plots
show_p2p = False
y_lim=[0, .85]

if show_p2p:
	nrows = 2
else:
	nrows = 1
ncols = 3
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 5))
for mod_iter in range(6):
	#axes[np.divmod(mod_iter,3)[0]][np.divmod(mod_iter,3)[1]].plot(mean_curve_all[mod_iter,:])
	if show_p2p:
		if mod_iter < 3:
			axes[0][0].errorbar(x=np.arange(number_of_refinements+1)+mod_iter/10, y=average_curve_mean_all[mod_iter,:], yerr=average_curve_std_all[mod_iter,:])
			axes[0][1].errorbar(x=np.arange(number_of_refinements+1)+mod_iter/10, y=q0_curve_mean_all[mod_iter,:], yerr=q0_curve_std_all[mod_iter,:])
			axes[0][2].errorbar(x=np.arange(number_of_refinements+1)+mod_iter/10, y=q1_curve_mean_all[mod_iter,:], yerr=q1_curve_std_all[mod_iter,:])
		else:
			axes[1][0].errorbar(x=np.arange(number_of_refinements+1)+mod_iter/10, y=average_curve_mean_all[mod_iter,:], yerr=average_curve_std_all[mod_iter,:])
			axes[1][1].errorbar(x=np.arange(number_of_refinements+1)+mod_iter/10, y=q0_curve_mean_all[mod_iter,:], yerr=q0_curve_std_all[mod_iter,:])
			axes[1][2].errorbar(x=np.arange(number_of_refinements+1)+mod_iter/10, y=q1_curve_mean_all[mod_iter,:], yerr=q1_curve_std_all[mod_iter,:])
	else:
		if mod_iter < 3:
			axes[0].errorbar(x=np.arange(number_of_refinements+1)+mod_iter/10, y=average_curve_mean_all[mod_iter,:], yerr=average_curve_std_all[mod_iter,:])
			axes[1].errorbar(x=np.arange(number_of_refinements+1)+mod_iter/10, y=q0_curve_mean_all[mod_iter,:], yerr=q0_curve_std_all[mod_iter,:])
			axes[2].errorbar(x=np.arange(number_of_refinements+1)+mod_iter/10, y=q1_curve_mean_all[mod_iter,:], yerr=q1_curve_std_all[mod_iter,:])

for subplot_iter in range(nrows*ncols):
	if show_p2p:
		axes[np.divmod(subplot_iter,3)[0]][np.divmod(subplot_iter,3)[1]].set_ylim(y_lim)
		axes[np.divmod(subplot_iter,3)[0]][np.divmod(subplot_iter,3)[1]].legend(['a','b','c'])
	else:
		axes[subplot_iter].set_ylim(y_lim)
plt.show()
#import pdb; pdb.set_trace()
#axes[0].


#import pdb; pdb.set_trace()
