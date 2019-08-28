import gym
from gym.wrappers import Monitor
import numpy as np

#from stable_baselines.common.policies import MlpPolicy
#from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import PPO2, DDPG

RL_method = "DDPG"

save_name_extension = RL_method+'test'
# defining the environments
env = gym.make('NmiLeg-v1')
#env = DummyVecEnv([lambda: env])

# loading the trained model
if RL_method == "PPO2":
	model = PPO2.load("trainedmodel-nmiLeg"+save_name_extension)
elif RL_method == "DDPG":
	model = DDPG.load("trainedmodel-nmiLeg"+save_name_extension)
else:
	raise ValueError("Invalid RL mode")
# setting the environment
model.set_env(env)
##
env_run = gym.make('NmiLeg-v1')
env_run = Monitor(env_run,'./video/'+save_name_extension,force=True)
#model = DDPG.load("PPO2-HalfCheetah_nssu-v3_test2")
obs = env_run.reset()
#while True:
run_length_seconds = 10
time_step = .01
run_samples = int(np.round(run_length_seconds/(time_step*5)))
print(run_samples)
rew_sum=0
for _ in range(run_samples):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env_run.step(action)
    rew_sum=rew_sum+rewards
    env_run.render()
print(rew_sum)
#import pdb; pdb.set_trace()

