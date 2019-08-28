
# next is to add accel and see the difference
# add stiffness too
import numpy as np
from matplotlib import pyplot as plt
from all_functions import *
import pickle
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)

experiment_ID = "transferability_test"
mc_run_number = 50
babbling_time = 5
errors_all_A_A = np.zeros([2, mc_run_number])
errors_all_A_B = np.zeros([2, mc_run_number])

stiffness_version = 0
MuJoCo_model_name_A="nmi_leg_w_chassis_air_v{}.xml".format(stiffness_version)
stiffness_version = 4
MuJoCo_model_name_B="nmi_leg_w_chassis_air_v{}.xml".format(stiffness_version)

np.random.seed(0) # change the seed for different initial conditions
tf.random.set_random_seed(0)
for mc_counter in range(mc_run_number):
	# train model_A
	[babbling_kinematics, babbling_activations] =\
		babbling_fcn(
			MuJoCo_model_name=MuJoCo_model_name_A,
			simulation_minutes=babbling_time,
			kinematics_activations_show=False)
	model_A = inverse_mapping_fcn(
		kinematics=babbling_kinematics,
		activations=babbling_activations,
		log_address="./logs/{}/A_A/scalars/mc_run{}".format(experiment_ID, mc_counter),
		early_stopping=False)
	cum_kinematics = babbling_kinematics
	cum_activations = babbling_activations
	#A_A test
	[model_A, errors, cum_kinematics, cum_activations] =\
		in_air_adaptation_fcn(
			MuJoCo_model_name=MuJoCo_model_name_A,
			model=model_A,
			babbling_kinematics=cum_kinematics,
			babbling_activations=cum_activations,
			number_of_refinements=0,
			log_address="./logs/{}/A_A/scalars/mc_run{}".format(experiment_ID, mc_counter),
			error_plots_show=False,
			Mj_render=False)
	import pdb; pdb.set_trace()
	errors_all_A_A[:, mc_counter] = [errors[0][0],errors[1][0]]
	#A_B test
	[model_A, errors, cum_kinematics, cum_activations] =\
		in_air_adaptation_fcn(
			MuJoCo_model_name=MuJoCo_model_name_B,
			model=model_A,
			babbling_kinematics=cum_kinematics,
			babbling_activations=cum_activations,
			number_of_refinements=0,
			error_plots_show=False,
			Mj_render=False)
	errors_all_A_B[:, mc_counter] = [errors[0][0],errors[1][0]]

## saving the results
np.save("./results/{}/errors_all_A_A".format(experiment_ID),errors_all_A_A)
np.save("./results/{}/errors_all_A_B".format(experiment_ID),errors_all_A_B)
## printing the results
print("errors_mean: ",errors_all_A_A.mean(1))
print("errors_std: ",errors_all_A_A.std(1))
print("errors_mean: ",errors_all_A_B.mean(1))
print("errors_std: ",errors_all_A_B.std(1))
#import pdb; pdb.set_trace()