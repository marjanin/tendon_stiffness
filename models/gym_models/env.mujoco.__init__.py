from gym.envs.mujoco.mujoco_env import MujocoEnv
# ^^^^^ so that user gets the correct error
# message if mujoco is not installed correctly
from gym.envs.mujoco.ant import AntEnv
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv
from gym.envs.mujoco.hopper import HopperEnv
from gym.envs.mujoco.walker2d import Walker2dEnv
from gym.envs.mujoco.humanoid import HumanoidEnv
from gym.envs.mujoco.inverted_pendulum import InvertedPendulumEnv
from gym.envs.mujoco.inverted_double_pendulum import InvertedDoublePendulumEnv
from gym.envs.mujoco.reacher import ReacherEnv
from gym.envs.mujoco.swimmer import SwimmerEnv
from gym.envs.mujoco.humanoidstandup import HumanoidStandupEnv
from gym.envs.mujoco.pusher import PusherEnv
from gym.envs.mujoco.thrower import ThrowerEnv
from gym.envs.mujoco.striker import StrikerEnv
## added by Ali
from gym.envs.mujoco.myenv import HalfCheetahEnv
from gym.envs.mujoco.half_cheetah_v3_notstiff_standup import HalfCheetahEnv
from gym.envs.mujoco.half_cheetah_v3_notstiff_run import HalfCheetahEnv
from gym.envs.mujoco.nmi_leg_v1 import nmiLegEnv
from gym.envs.mujoco.TSNMILegv1 import TSNMILeg0Env, TSNMILeg1Env, TSNMILeg2Env, TSNMILeg3Env, TSNMILeg4Env, TSNMILeg5Env, TSNMILeg6Env, TSNMILeg7Env, TSNMILeg8Env


##