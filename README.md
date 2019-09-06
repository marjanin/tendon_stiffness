# tendon_stiffness
To study the effects of tendon stiffness in autonomous control of tendon-driven systems

test1:
stiffness_versions = 7 #[0, 500, 1000, 2000, 4000, 7000, 10000]
mc_run_number = 100
babbling_times = [2]#np.arange(1,1+5)
errors_all = np.zeros([2, mc_run_number, len(babbling_times), stiffness_versions])

test2:
stiffness_versions = 7 #[0, 500, 1000, 2000, 4000, 7000, 10000]
mc_run_number = 50
babbling_times = [1]#np.arange(1,1+5)


experiment_ID = "transferability_1"
mc_run_number = 100
babbling_time = 2
errors_all_A_A = np.zeros([2, mc_run_number])
errors_all_A_B = np.zeros([2, mc_run_number])

experiment_3_pool:
"experiment_3_pool"
	energy_cost_weight = 0
	reward_thresh = 7
"experiment_3_pool_B"
	energy_cost_weight = 0
	reward_thresh = 5
"experiment_3_pool_C"
	energy_cost_weight = 0
	reward_thresh = 6
"experiment_3_pool_C"
	energy_cost_weight = 0
	reward_thresh = 5
	mc_run_number = 100
"experiment_3_pool_?"
	energy_cost_weight = 0.001
	reward_thresh = 5
	mc_run_number = 50
"experiment_3_pool_E"
	energy_cost_weight = 0
	xml friction and damping of the chassis changes
	reward_thresh = 4
	mc_run_number = 20
	xml friction and damping of the chassis changes
	damping 50 friction 10 (used to be 0, 30)
"experiment_3_pool_F"
	energy_cost_weight = 0
	xml friction and damping of the chassis changes
	reward_thresh = 3
	mc_run_number = 20
	xml friction and damping of the chassis changes
	damping 50 friction 10 (used to be 0, 30)
"experiment_3_pool_G"
	energy_cost_weight = 0
	xml friction and damping of the chassis changes
	reward_thresh = 3
	mc_run_number = 100
	xml friction and damping of the chassis changes
	damping 50 friction 10 (used to be 0, 30)