
# next is to add accel and see the difference
# add stiffness too
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from all_functions import *
import pickle
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)

experiment_ID = "rl_transfer_learning_1"

mc_run_number = 1
babbling_time = 3
number_of_refinements = 0
errors_all_A_A = np.zeros([2, number_of_refinements+1, mc_run_number])
errors_all_A_B = np.zeros([2, number_of_refinements+1, mc_run_number])
errors_all_B_B = np.zeros([2, number_of_refinements+1, mc_run_number])

stiffness_version_A = 8
MuJoCo_model_name_A="nmi_leg_w_chassis_air_v{}.xml".format(stiffness_version_A)
MuJoCo_model_name_A_walk="nmi_leg_w_chassis_air_v{}_walk.xml".format(stiffness_version_A)
stiffness_version_B = 3
MuJoCo_model_name_B="nmi_leg_w_chassis_air_v{}.xml".format(stiffness_version_B)
MuJoCo_model_name_B_walk="nmi_leg_w_chassis_air_v{}_walk.xml".format(stiffness_version_B)

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
	model_A_refined_A=tf.keras.models.clone_model(model_A_babble)
	model_A_refined_A.compile(optimizer=tf.train.AdamOptimizer(0.01),
          loss='mse',       # mean squared error
          metrics=['mse'])  # mean squared error
	[model_A_refined_A, errors, cum_kinematics_A_refined, cum_activations_A_refined] =\
		in_air_adaptation_fcn(
			MuJoCo_model_name=MuJoCo_model_name_A,
			model=model_A_refined_A,
			babbling_kinematics=cum_kinematics_A_babble,
			babbling_activations=cum_activations_A_babble,
			number_of_refinements=number_of_refinements,
			log_address="./logs/{}/scalars/refine_A_A_mc_run{}/".format(experiment_ID, mc_counter),
			error_plots_show=False,
			Mj_render=False)
	#import pdb; pdb.set_trace()
	errors_all_A_A[:,:, mc_counter] = [errors[0],errors[1]]
	#A_B test
	np.random.seed(random_seed) # change the seed for different initial conditions
	tf.random.set_random_seed(random_seed)
	model_A_refined_B=tf.keras.models.clone_model(model_A_babble)
	model_A_refined_B.compile(optimizer=tf.train.AdamOptimizer(0.01),
      loss='mse',       # mean squared error
      metrics=['mse'])  # mean squared error
	[model_A_refined_B, errors, cum_kinematics_B_refined, cum_activations_B_refined] =\
		in_air_adaptation_fcn(
			MuJoCo_model_name=MuJoCo_model_name_B,
			model=model_A_refined_B,
			babbling_kinematics=cum_kinematics_A_babble,
			babbling_activations=cum_activations_A_babble,
			number_of_refinements=number_of_refinements,
			log_address="./logs/{}/scalars/refine_A_B_mc_run{}/".format(experiment_ID, mc_counter),
			error_plots_show=False,
			Mj_render=False)
	errors_all_A_B[:,:, mc_counter] = [errors[0],errors[1]]
## saving the results
np.save("./results/{}/errors_all_A_A".format(experiment_ID),errors_all_A_A)
np.save("./results/{}/errors_all_A_B".format(experiment_ID),errors_all_A_B)
np.save("./results/{}/errors_all_B_B".format(experiment_ID),errors_all_B_B)

np.random.seed(random_seed+2) # change the seed for different initial conditions
tf.random.set_random_seed(random_seed+2)
## Higher level learning (RL)
[ best_reward_so_far, all_rewards, best_features_so_far, real_attempt_activations ]=\
 learn_to_move_fcn(
 	MuJoCo_model_name=MuJoCo_model_name_A_walk,
 	model=model_A_refined_A,
 	cum_kinematics=cum_kinematics_A_babble,
 	cum_activations=cum_activations_A_babble,
 	reward_thresh=6,
 	refinement=False,
 	Mj_render=False)
#import pdb; pdb.set_trace()
[rewardA_A, _, _, _, _]=\
feat_to_run_attempt_fcn(MuJoCo_model_name=MuJoCo_model_name_A_walk, features=best_features_so_far, model=model_A_refined_A, feat_show=True, Mj_render=True)
[rewardA_B, _, _, _, _]=\
feat_to_run_attempt_fcn(MuJoCo_model_name=MuJoCo_model_name_B_walk, features=best_features_so_far, model=model_A_refined_A, feat_show=True, Mj_render=False)
[rewardAB_B, _, _, _, _]=\
feat_to_run_attempt_fcn(MuJoCo_model_name=MuJoCo_model_name_B_walk, features=best_features_so_far, model=model_A_refined_B, feat_show=True, Mj_render=False)
print("rewardA_A: ", rewardA_A)
print("rewardA_B: ", rewardA_B)
print("rewardAB_B: ", rewardAB_B)


## printing the results
# print("errors_mean: ",errors_all_A_A.mean(1))
# print("errors_std: ",errors_all_A_A.std(1))
# print("errors_mean: ",errors_all_A_B.mean(1))
# print("errors_std: ",errors_all_A_B.std(1))
#import pdb; pdb.set_trace()
