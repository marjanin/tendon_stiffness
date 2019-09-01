
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
stiffness_versions = 9#[0, 500, 1000, 2000, 4000, 7000, 10000. 15000, 20000]
mc_run_number = 50
babbling_times = [3]#np.arange(1,1+5)
epoch_numbers = 20
histories = np.empty([mc_run_number, len(babbling_times), stiffness_versions]).tolist()
learning_errors = np.zeros([mc_run_number, len(babbling_times), stiffness_versions, epoch_numbers])
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 4.5))
for stiffness_ver in range(stiffness_versions):
	MuJoCo_model_name="nmi_leg_w_chassis_air_v{}.xml".format(stiffness_ver)
	for babbling_time_cntr in range(len(babbling_times)):
		for mc_counter in range(mc_run_number):
			logdir="./logs/{}/scalars/stiffness_v{}/babbling_time_{}_mins/mc_run{}".format(experiment_ID, stiffness_ver,babbling_times[babbling_time_cntr], mc_counter)
			with open(logdir+'/trainHistoryDict.pickle', 'rb') as file_pi:
				history=pickle.load(file_pi)
			histories[mc_counter][babbling_time_cntr][stiffness_ver] = history
			learning_errors[mc_counter, babbling_time_cntr, stiffness_ver, :] = np.array(history['loss'])
learning_errors_per_stiffness = np.zeros([stiffness_versions, epoch_numbers])
for stiffness_ver in range(stiffness_versions):
	learning_errors_per_stiffness[stiffness_ver,:] = learning_errors[:, :, stiffness_ver, :].mean(0).squeeze()
	axes.plot(np.arange(1,epoch_numbers+1), learning_errors_per_stiffness[stiffness_ver,:])
axes.legend(['S: 0', 'S: 500', 'S: 1K', 'S: 2K', 'S: 4K', 'S: 7K', 'S: 10K', 'S: 15K', 'S: 20K'])
#(50, 1, 9, 20) 
axes.set_xlabel("epoch #")
axes.set_ylabel("epoch MSE")
axes.set_title("learning curves for different stiffness values (S)")
fig.savefig('./results/{}/learningcurves.png'.format(experiment_ID))
plt.show()



#import pdb; pdb.set_trace()
