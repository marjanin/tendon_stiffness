experiment_ID = "transfer_learning_6"
mc_run_number = 50
babbling_time = 3
number_of_refinements = 5
errors_all_A_A = np.zeros([2, number_of_refinements+1, mc_run_number])
errors_all_A_B = np.zeros([2, number_of_refinements+1, mc_run_number])
errors_all_B_B = np.zeros([2, number_of_refinements+1, mc_run_number])

stiffness_version_A = 5
MuJoCo_model_name_A="nmi_leg_w_chassis_air_v{}.xml".format(stiffness_version_A)
stiffness_version_B = 2
MuJoCo_model_name_B="nmi_leg_w_chassis_air_v{}.xml".format(stiffness_version_B)