experiment_ID="error_vs_stiffness_test5_cyclical_p2p"
stiffness_versions = 9 #[0, 500, 1000, 2000, 4000, 7000, 10000. 15000, 20000]
mc_run_number = 50
babbling_times = [3]#np.arange(1,1+5)
errors_all_cyclical = np.zeros([2, mc_run_number, len(babbling_times), stiffness_versions])
errors_all_p2p = np.zeros([2, mc_run_number, len(babbling_times), stiffness_versions])