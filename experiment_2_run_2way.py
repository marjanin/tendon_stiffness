
# next is to add accel and see the difference
# add stiffness too
import numpy as np
from matplotlib import pyplot as plt
from all_functions import *
import pickle
import os
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)

experiment_ID = "experiment_2_2way_"
mc_run_number = 50
babbling_time = 3
number_of_refinements = 5
errors_all_cyc_A_A = np.zeros([2, number_of_refinements+1, mc_run_number])
errors_all_cyc_A_B = np.zeros([2, number_of_refinements+1, mc_run_number])
errors_all_cyc_B_B = np.zeros([2, number_of_refinements+1, mc_run_number])
errors_all_cyc_B_A = np.zeros([2, number_of_refinements+1, mc_run_number])

errors_all_p2p_A_A = np.zeros([2, number_of_refinements+1, mc_run_number])
errors_all_p2p_A_B = np.zeros([2, number_of_refinements+1, mc_run_number])
errors_all_p2p_B_B = np.zeros([2, number_of_refinements+1, mc_run_number])
errors_all_p2p_B_A = np.zeros([2, number_of_refinements+1, mc_run_number])

stiffness_version_A = 5
MuJoCo_model_name_A="nmi_leg_w_chassis_air_v{}.xml".format(stiffness_version_A)
stiffness_version_B = 2
MuJoCo_model_name_B="nmi_leg_w_chassis_air_v{}.xml".format(stiffness_version_B)

random_seed = -1

for mc_counter in range(mc_run_number):
	random_seed+=1
	# train model_A
	np.random.seed(random_seed) # change the seed for different initial conditions
	tf.random.set_random_seed(random_seed)
	[babbling_kinematics, babbling_activations] =\
		babbling_fcn(
			MuJoCo_model_name=MuJoCo_model_name_A,
			simulation_minutes=babbling_time,
			kinematics_activations_show=False)
	model_A_babble = inverse_mapping_fcn(
		kinematics=babbling_kinematics,
		activations=babbling_activations,
		log_address="./logs/{}/scalars/babble_A_mc_run{}/".format(experiment_ID, mc_counter),
		early_stopping=False)
	cum_kinematics_A_babble = babbling_kinematics
	cum_activations_A_babble = babbling_activations
	#A_A test
	np.random.seed(random_seed) # change the seed for different initial conditions
	tf.random.set_random_seed(random_seed)
	model_A_refined_A_cyc = copy_model_fcn(original_model=model_A_babble)
	[model_A_refined_A_cyc, errors, cum_kinematics_A_refined, cum_activations_A_refined] =\
		in_air_adaptation_fcn(
			MuJoCo_model_name=MuJoCo_model_name_A,
			model=model_A_refined_A_cyc,
			babbling_kinematics=cum_kinematics_A_babble,
			babbling_activations=cum_activations_A_babble,
			number_of_refinements=number_of_refinements,
			log_address="./logs/{}/scalars/refine_cyc_A_A_mc_run{}/".format(experiment_ID, mc_counter),
			error_plots_show=False,
			Mj_render=False)
	errors_all_cyc_A_A[:,:, mc_counter] = [errors[0],errors[1]]

	np.random.seed(random_seed) # change the seed for different initial conditions
	tf.random.set_random_seed(random_seed)
	model_A_refined_A_p2p = copy_model_fcn(original_model=model_A_babble)
	[model_A_refined_A_p2p, errors, cum_kinematics_A_refined, cum_activations_A_refined] =\
		p2p_run_fcn(
			MuJoCo_model_name=MuJoCo_model_name_A,
			model=model_A_refined_A_p2p,
			babbling_kinematics=cum_kinematics_A_babble,
			babbling_activations=cum_activations_A_babble,
			number_of_refinements=number_of_refinements,
			log_address="./logs/{}/scalars/refine_p2p_A_A_mc_run{}/".format(experiment_ID, mc_counter),
			error_plots_show=False,
			Mj_render=False)
	errors_all_p2p_A_A[:,:, mc_counter] = [errors[0],errors[1]]
	#A_B test
	np.random.seed(random_seed) # change the seed for different initial conditions
	tf.random.set_random_seed(random_seed)
	model_A_refined_B_cyc = copy_model_fcn(original_model=model_A_babble)
	[model_A_refined_B_cyc, errors, cum_kinematics_B_refined, cum_activations_B_refined] =\
		in_air_adaptation_fcn(
			MuJoCo_model_name=MuJoCo_model_name_B,
			model=model_A_refined_B_cyc,
			babbling_kinematics=cum_kinematics_A_babble,
			babbling_activations=cum_activations_A_babble,
			number_of_refinements=number_of_refinements,
			log_address="./logs/{}/scalars/refine_cyc_A_B_mc_run{}/".format(experiment_ID, mc_counter),
			error_plots_show=False,
			Mj_render=False)
	errors_all_cyc_A_B[:,:, mc_counter] = [errors[0],errors[1]]

	np.random.seed(random_seed) # change the seed for different initial conditions
	tf.random.set_random_seed(random_seed)
	model_A_refined_B_p2p = copy_model_fcn(original_model=model_A_babble)
	[model_A_refined_B_p2p, errors, cum_kinematics_B_refined, cum_activations_B_refined] =\
		p2p_run_fcn(
			MuJoCo_model_name=MuJoCo_model_name_B,
			model=model_A_refined_B_p2p,
			babbling_kinematics=cum_kinematics_A_babble,
			babbling_activations=cum_activations_A_babble,
			number_of_refinements=number_of_refinements,
			log_address="./logs/{}/scalars/refine_p2p_A_B_mc_run{}/".format(experiment_ID, mc_counter),
			error_plots_show=False,
			Mj_render=False)
	errors_all_p2p_A_B[:,:, mc_counter] = [errors[0],errors[1]]

	# train model_B
	np.random.seed(random_seed) # change the seed for different initial conditions
	tf.random.set_random_seed(random_seed)
	[babbling_kinematics, babbling_activations] =\
		babbling_fcn(
			MuJoCo_model_name=MuJoCo_model_name_B,
			simulation_minutes=babbling_time,
			kinematics_activations_show=False)
	model_B_babble = inverse_mapping_fcn(
		kinematics=babbling_kinematics,
		activations=babbling_activations,
		log_address="./logs/{}/scalars/babble_B_mc_run{}/".format(experiment_ID, mc_counter),
		early_stopping=False)
	cum_kinematics_B_babble = babbling_kinematics
	cum_activations_B_babble = babbling_activations
	#B_B test
	np.random.seed(random_seed) # change the seed for different initial conditions
	tf.random.set_random_seed(random_seed)
	model_B_refined_B_cyc = copy_model_fcn(original_model=model_B_babble)
	[model_B_refined_B_cyc, errors, cum_kinematics_B_refined, cum_activations_B_refined] =\
		in_air_adaptation_fcn(
			MuJoCo_model_name=MuJoCo_model_name_B,
			model=model_B_refined_B_cyc,
			babbling_kinematics=cum_kinematics_B_babble,
			babbling_activations=cum_activations_B_babble,
			number_of_refinements=number_of_refinements,
			log_address="./logs/{}/scalars/refine_cyc_B_B_mc_run{}/".format(experiment_ID, mc_counter),
			error_plots_show=False,
			Mj_render=False)
	errors_all_cyc_B_B[:,:, mc_counter] = [errors[0],errors[1]]

	np.random.seed(random_seed) # change the seed for different initial conditions
	tf.random.set_random_seed(random_seed)
	model_B_refined_B_p2p = copy_model_fcn(original_model=model_B_babble)
	[model_B_refined_B_p2p, errors, cum_kinematics_B_refined, cum_activations_B_refined] =\
		p2p_run_fcn(
			MuJoCo_model_name=MuJoCo_model_name_B,
			model=model_B_refined_B_p2p,
			babbling_kinematics=cum_kinematics_B_babble,
			babbling_activations=cum_activations_B_babble,
			number_of_refinements=number_of_refinements,
			log_address="./logs/{}/scalars/refine_p2p_B_B_mc_run{}/".format(experiment_ID, mc_counter),
			error_plots_show=False,
			Mj_render=False)
	errors_all_p2p_B_B[:,:, mc_counter] = [errors[0],errors[1]]

	#B_A test
	np.random.seed(random_seed) # change the seed for different initial conditions
	tf.random.set_random_seed(random_seed)
	model_B_refined_A_cyc = copy_model_fcn(original_model=model_B_babble)
	[model_B_refined_A_cyc, errors, cum_kinematics_A_refined, cum_activations_A_refined] =\
		in_air_adaptation_fcn(
			MuJoCo_model_name=MuJoCo_model_name_A,
			model=model_B_refined_A_cyc,
			babbling_kinematics=cum_kinematics_B_babble,
			babbling_activations=cum_activations_B_babble,
			number_of_refinements=number_of_refinements,
			log_address="./logs/{}/scalars/refine_cyc_B_A_mc_run{}/".format(experiment_ID, mc_counter),
			error_plots_show=False,
			Mj_render=False)
	errors_all_cyc_B_A[:,:, mc_counter] = [errors[0],errors[1]]

	np.random.seed(random_seed) # change the seed for different initial conditions
	tf.random.set_random_seed(random_seed)
	model_B_refined_A_p2p = copy_model_fcn(original_model=model_B_babble)
	[model_B_refined_A_p2p, errors, cum_kinematics_A_refined, cum_activations_A_refined] =\
		p2p_run_fcn(
			MuJoCo_model_name=MuJoCo_model_name_A,
			model=model_B_refined_A_cyc,
			babbling_kinematics=cum_kinematics_B_babble,
			babbling_activations=cum_activations_B_babble,
			number_of_refinements=number_of_refinements,
			log_address="./logs/{}/scalars/refine_p2p_B_A_mc_run{}/".format(experiment_ID, mc_counter),
			error_plots_show=False,
			Mj_render=False)
	errors_all_p2p_B_A[:,:, mc_counter] = [errors[0],errors[1]]

## saving the results
os.makedirs("./results/{}".format(experiment_ID), exist_ok=True)
np.save("./results/{}/errors_all_cyc_A_A".format(experiment_ID),errors_all_cyc_A_A)
np.save("./results/{}/errors_all_cyc_A_B".format(experiment_ID),errors_all_cyc_A_B)
np.save("./results/{}/errors_all_cyc_B_B".format(experiment_ID),errors_all_cyc_B_B)
np.save("./results/{}/errors_all_cyc_B_A".format(experiment_ID),errors_all_cyc_B_A)

np.save("./results/{}/errors_all_p2p_A_A".format(experiment_ID),errors_all_p2p_A_A)
np.save("./results/{}/errors_all_p2p_A_B".format(experiment_ID),errors_all_p2p_A_B)
np.save("./results/{}/errors_all_p2p_B_B".format(experiment_ID),errors_all_p2p_B_B)
np.save("./results/{}/errors_all_p2p_B_A".format(experiment_ID),errors_all_p2p_B_A)

## printing the results
# print("errors_mean: ",errors_all_A_A.mean(1))
# print("errors_std: ",errors_all_A_A.std(1))
# print("errors_mean: ",errors_all_A_B.mean(1))
# print("errors_std: ",errors_all_A_B.std(1))
#import pdb; pdb.set_trace()
