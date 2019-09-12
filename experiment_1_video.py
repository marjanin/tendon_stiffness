
# next is to add accel and see the difference
# add stiffness too
import numpy as np
from matplotlib import pyplot as plt
from all_functions import *
import pickle
import os
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)
Mj_render=True
experiment_ID="experiment_1B_"
stiffness_versions = 1#[0, 500, 1000, 2000, 4000, 7000, 10000. 15000, 20000]
mc_run_number = 1
babbling_times = [3]#np.arange(1,1+5)
errors_all_cyclical = np.zeros([2, mc_run_number, len(babbling_times), stiffness_versions])
errors_all_p2p = np.zeros([2, mc_run_number, len(babbling_times), stiffness_versions])
for stiffness_ver in [6]:
	MuJoCo_model_name="nmi_leg_w_chassis_air_v{}.xml".format(stiffness_ver)
	for babbling_time_cntr in range(len(babbling_times)):
		np.random.seed(0) # change the seed for different initial conditions
		tf.random.set_random_seed(0)
		for mc_counter in range(mc_run_number):
			[babbling_kinematics, babbling_activations] =\
				babbling_fcn(
					MuJoCo_model_name=MuJoCo_model_name,
					simulation_minutes=babbling_times[babbling_time_cntr],
					kinematics_activations_show=False)
			model = \
			inverse_mapping_fcn(
				kinematics=babbling_kinematics,
				activations=babbling_activations,
				log_address="./logs/{}/scalars/stiffness_v{}/babbling_time_{}_mins/mc_run{}".format(experiment_ID, stiffness_ver,babbling_times[babbling_time_cntr], mc_counter),
				early_stopping=False)
			cum_kinematics = babbling_kinematics
			cum_activations = babbling_activations
			#np.random.seed(0) # change the seed for different initial conditions
			[_, errors, cum_kinematics, cum_activations] =\
				in_air_adaptation_fcn(
					MuJoCo_model_name=MuJoCo_model_name,
					model=model,
					babbling_kinematics=cum_kinematics,
					babbling_activations=cum_activations,
					number_of_refinements=0,
					error_plots_show=False,
					Mj_render=Mj_render)
			#errors_all_cyclical[:, mc_counter, babbling_time_cntr, stiffness_ver] = [errors[0][0],errors[1][0]]
			
			[_, errors, cum_kinematics, cum_activations] =\
				p2p_run_fcn(
					MuJoCo_model_name=MuJoCo_model_name,
					model=model,
					babbling_kinematics=cum_kinematics,
					babbling_activations=cum_activations,
					number_of_refinements=0,
					error_plots_show=False,
					Mj_render=Mj_render)
			#errors_all_p2p[:, mc_counter, babbling_time_cntr, stiffness_ver] = [errors[0][0],errors[1][0]]

# os.makedirs("./results/{}".format(experiment_ID), exist_ok=True)
# print("errors_mean: ",errors_all_cyclical.mean(1))
# print("errors_std: ",errors_all_cyclical.std(1))
# np.save("./results/{}/errors_all_cyclical".format(experiment_ID),errors_all_cyclical)
# print("errors_mean: ",errors_all_p2p.mean(1))
# print("errors_std: ",errors_all_p2p.std(1))
# np.save("./results/{}/errors_all_p2p".format(experiment_ID),errors_all_p2p)
#import pdb; pdb.set_trace()

