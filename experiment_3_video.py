
# next is to add accel and see the difference
# add stiffness too
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from all_functions import *
import pickle
from warnings import simplefilter
import multiprocessing as mp


simplefilter(action='ignore', category=FutureWarning)
['S0: 0', 'S1: 500', 'S2: 1K', 'S3: 2K', 'S4: 4K', 'S5: 7K', 'S6: 10K', 'S7: 15K', 'S8: 20K']
#stiffness_version_A=5
experiment_ID = "experiment_3_video"
reward_thresh = 3 #3:7 3B:5
mc_run_number = 1
babbling_time = 3
number_of_refinements = 0

rewards = np.zeros([mc_run_number])
energies = np.zeros([mc_run_number])
exploration_run_numbers = np.zeros([mc_run_number])
#[0, 2, 4, 5, 7]
for stiffness_version_A in [0, 3, 5, 8]:
	MuJoCo_model_name_A="nmi_leg_w_chassis_air_v{}.xml".format(stiffness_version_A)
	MuJoCo_model_name_A_walk="nmi_leg_w_chassis_air_v{}_walk.xml".format(stiffness_version_A)
	if stiffness_version_A==0 or stiffness_version_A==5:
		random_seed = 4
	else:
		random_seed = 0
	random_seed=-1
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
			log_address="./logs/{}/scalars/stiffness_version{}/babble_A_mc_run{}/".format(experiment_ID, stiffness_version_A, mc_counter),
			early_stopping=False)
		cum_kinematics_A_babble = babbling_kinematics
		cum_activations_A_babble = babbling_activations
		#A_A test
		np.random.seed(random_seed) # change the seed for different initial conditions
		tf.random.set_random_seed(random_seed)
		[ best_reward_so_far, all_rewards, best_features_so_far, real_attempt_activations, exploration_run_no ]=\
		learn_to_move_2_fcn(
			MuJoCo_model_name=MuJoCo_model_name_A_walk,
			model=model_A_babble,
			cum_kinematics=cum_kinematics_A_babble,
			cum_activations=cum_activations_A_babble,
			reward_thresh=reward_thresh,
			energy_cost_weight=0,
			refinement=False,
			Mj_render=False)
		[rewardA_A, _, _, _, real_attempt_activations]=\
		feat_to_run_attempt_fcn(
			MuJoCo_model_name=MuJoCo_model_name_A_walk,
			features=best_features_so_far,
			model=model_A_babble,
			feat_show=False,
			Mj_render=True)
		total_energyA_A = np.square(real_attempt_activations).sum(0).sum(0)

		print("traveled distance: ", rewardA_A)
		print("consumed energy: ", total_energyA_A)
		exploration_run_numbers[mc_counter] = exploration_run_no
		rewards[mc_counter] = rewardA_A
		energies[mc_counter] = total_energyA_A

# os.makedirs("./results/{}".format(experiment_ID), exist_ok=True)
# np.save("./results/{}/exploration_run_numbers_S{}".format(experiment_ID, stiffness_version_A),exploration_run_numbers)
# np.save("./results/{}/rewards_S{}".format(experiment_ID, stiffness_version_A),rewards)
# np.save("./results/{}/energies_S{}".format(experiment_ID, stiffness_version_A),energies)




#print("best_reward_so_far: ", best_reward_so_far)
## printing the results
# print("errors_mean: ",errors_all_A_A.mean(1))
# print("errors_std: ",errors_all_A_A.std(1))
# print("errors_mean: ",errors_all_A_B.mean(1))
# print("errors_std: ",errors_all_A_B.std(1))

# import pdb; pdb.set_trace()
