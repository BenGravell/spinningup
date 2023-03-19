from spinup.algos.pytorch.vpg.vpg_ben import vpg
from spinup.algos.pytorch.ppo.ppo_ben import ppo

# from spinup.algos.pytorch.trpo.trpo_ben import trpo
from spinup.algos.tf1.trpo.trpo import trpo

import gym


# env_fn = lambda: gym.make('CartPole-v0')
# logger_kwargs = dict(output_dir='C:\\Users\\bjgra\\Documents\\GitHub\\spinningup\\data\\algo_ben\\cartpole',
#                      exp_name='algo_cartpole')
# # actor-critic kwargs
# ac_kwargs = dict(hidden_sizes=[32, 32])

env_fn = lambda: gym.make('gym_ballbeam:BallBeam-v0')
# env_fn = lambda: gym.make('gym_ballbeam:BallBeamDiscrete-v0')
logger_kwargs = dict(output_dir='C:\\Users\\bjgra\\Documents\\GitHub\\spinningup\\data\\algo_ben\\ballbeam',
                     exp_name='algo_ballbeam')

# actor-critic kwargs
ac_kwargs = dict(hidden_sizes=[16])

# algo = vpg
# algo = ppo
algo = trpo

algo(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=3*200, epochs=20, seed=0, logger_kwargs=logger_kwargs)

# run this from command prompt to view the trained policy:
# python -m spinup.run test_policy C:\Users\bjgra\Documents\GitHub\spinningup\data\algo_ben\cartpole
# python -m spinup.run test_policy C:\Users\bjgra\Documents\GitHub\spinningup\data\algo_ben\ballbeam
