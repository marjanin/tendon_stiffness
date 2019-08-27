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


