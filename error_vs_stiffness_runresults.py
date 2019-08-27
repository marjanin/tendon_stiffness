
# next is to add accel and see the difference
# add stiffness too
import numpy as np
from scipy import signal, stats
from matplotlib import pyplot as plt
from all_functions import *
import pickle
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

test_number = 3
file_address = "./results/errors_all_test{}.npy".format(test_number)

errors_all = np.load(file_address)
errors_all_mean=errors_all.mean(1)
errors_all_std = errors_all.std(1)
print("errors_mean: ",errors_all_mean)
print("errors_std: ",errors_all_std)
# import pdb; pdb.set_trace()
# Error bars
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 5))

N = 7
stiffness_values = ["0", "500", "1K", "2K", "4K", "7K", "10K"]
x = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence
#import pdb; pdb.set_trace()
#p1 = axes[0].bar(x, errors_all_mean[0,:,:][0], yerr=errors_all_std[0,:,:][0],align='center', alpha=0.5, ecolor='black', capsize=10)
#p2 = axes[1].bar(x, errors_all_mean[0,:,:][0], yerr=errors_all_std[0,:,:][0], alpha=0.5, ecolor='black', capsize=10)
#p1 = axes[0].boxplot([errors_all[0,:,0,0], errors_all[0,:,0,1], errors_all[0,:,0,2], errors_all[0,:,0,3], errors_all[0,:,0,4], errors_all[0,:,0,5], errors_all[0,:,0,6]])
p0 = axes[0].boxplot(
	errors_all.mean(0).reshape(1,-1,1,N).squeeze(),
	notch=True,
	patch_artist=True)
axes[0].set_title(r'$(q_0+q_1)/2$',fontsize=12)
axes[0].set_ylim([0, .75])
axes[0].set_xlabel('stiffness')
axes[0].set_xticklabels(stiffness_values, rotation=45, fontsize=8)
axes[0].set_ylabel('RMSE')
p1 = axes[1].boxplot(
	errors_all[0,:,:,:].reshape(1,-1,1,N).squeeze(),
	notch=True,
	patch_artist=True)
axes[1].set_title('$q_0$', fontsize=12)
axes[1].set_ylim([0, .75])
axes[1].set_yticklabels([])
axes[1].set_xlabel('stiffness')
axes[1].set_xticklabels(stiffness_values, rotation=45, fontsize=8)
p2 = axes[2].boxplot(
	errors_all[1,:,:,:].reshape(1,-1,1,N).squeeze(),
	notch=True,
	patch_artist=True)
axes[2].set_title('$q_1$', fontsize=12)
axes[2].set_ylim([0, .75])
axes[2].set_yticklabels([])
axes[2].set_xlabel('stiffness')
axes[2].set_xticklabels(stiffness_values, rotation=45, fontsize=8)


# p1 = axes[0].plt.bar(ind, errors_all_mean[0,:,:][0], yerr=errors_all_std[0,:,:][0])
# p2 = axes[1].plt.bar(ind+.5, errors_all_mean[1,:,:][0], yerr=errors_all_std[1,:,:][0])
#import pdb; pdb.set_trace()
[f_ow, p_val] = stats.f_oneway(errors_all.mean(0)[:,0,0],errors_all.mean(0)[:,0,2])
print("p-value: ", p_val)
plt.show()

# stiffness_versions = 7 #[0, 500, 1000, 2000, 4000, 7000, 10000]
# mc_run_number = 100
# babbling_times = [2]#np.arange(1,1+5)
# errors_all = np.zeros([2, mc_run_number, len(babbling_times), stiffness_versions])
# for stiffness_ver in range(stiffness_versions):
# 	MuJoCo_model_name="nmi_leg_w_chassis_air_v{}.xml".format(stiffness_ver)
# 	for babbling_time_cntr in range(len(babbling_times)):
# 		np.random.seed(0) # change the seed for different initial conditions
# 		tf.random.set_random_seed(0)
# 		for mc_counter in range(mc_run_number):
# 			[babbling_kinematics, babbling_activations] =\
# 				babbling_fcn(
# 					MuJoCo_model_name=MuJoCo_model_name,
# 					simulation_minutes=babbling_times[babbling_time_cntr],
# 					kinematics_activations_show=False)
# 			model = inverse_mapping_fcn(
# 				kinematics=babbling_kinematics,
# 				activations=babbling_activations,
# 				log_address="./logs/scalars/stiffness_v{}/babbling_time_{}_mins/mc_run{}".format(stiffness_ver,babbling_times[babbling_time_cntr], mc_counter),
# 				early_stopping=False)
# 			cum_kinematics = babbling_kinematics
# 			cum_activations = babbling_activations
# 			#np.random.seed(0) # change the seed for different initial conditions
# 			[model, errors, cum_kinematics, cum_activations] =\
# 				in_air_adaptation_fcn(
# 					MuJoCo_model_name=MuJoCo_model_name,
# 					model=model,
# 					babbling_kinematics=cum_kinematics,
# 					babbling_activations=cum_activations,
# 					number_of_refinements=0,
# 					error_plots_show=False,
# 					Mj_render=False)
# 			errors_all[:, mc_counter, babbling_time_cntr, stiffness_ver] = [errors[0][0],errors[1][0]]
# print("errors_mean: ",errors_all.mean(1))
# print("errors_std: ",errors_all.std(1))
#np.save("errors_all",errors_all)
#import pdb; pdb.set_trace()
