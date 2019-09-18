import os
import pickle

from envs import make_env

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

cuda = True

# create and keep best actors
envs = []
infos = dict()
critics = dict()
env_params = dict()

print('Loading models')
models_path = 'trained_models/180406'
for fn in os.listdir(models_path):
    model_path = os.path.join(models_path, fn)
    env_id = int(fn.split('_')[1])
    actor_critic, params, info = torch.load(model_path)
    if env_id not in infos or info['return'] > infos[env_id]['return']:
        infos[env_id] = info
        env_params[env_id] = params
        if cuda:
            critics[env_id] = actor_critic.eval().cuda()
        else:
            critics[env_id] = actor_critic.eval()

for env_id, value in sorted(infos.items()):
    envs.append(make_env('YumiReacher-v0', env_id, 0, '/tmp', **env_params[env_id]))

print('Creating and resetting environments')
envs = SubprocVecEnv(envs)
dataset = []


n_timesteps = 250
cpu_actions = np.zeros((len(critics), envs.action_space.shape[0]))
obs = np.zeros((n_timesteps + 1, len(critics), envs.observation_space.shape[0]))
actions = np.zeros((n_timesteps, len(critics), envs.action_space.shape[0]))
rewards = np.zeros((n_timesteps, len(critics), 1))

n_rollouts = 1000

with open('value_dataset.pkl', 'wb') as f:
    for n in range(n_rollouts):
        print('Starting rollout {}/{}'.format(n + 1, n_rollouts))
        # rollout
        dataset = []
        obs[0] = o = envs.reset()
        for t in range(n_timesteps):
            if t % 100 == 0:
                print('Timestep {}/{}'.format(t, n_timesteps))
            inputs = Variable(torch.cuda.FloatTensor(o))
            for env_id, actor in critics.items():
                value, u, _, _ = actor.act(inputs[env_id:env_id + 1, :], None, None, deterministic=True)
                cpu_actions[env_id, :] = u.data.cpu().numpy()
            o, r, d, i = envs.step(cpu_actions)
            obs[t + 1] = o
            rewards[t, :, 0] = r
            actions[t] = cpu_actions
            
        # calculate values
        gamma = 0.99
        values = np.zeros((n_timesteps - 1, len(critics), 1))
        values[-1] = rewards[-1] / (1 - gamma)
        for t in reversed(range(n_timesteps - 2)):
            values[t] = rewards[t] + gamma * values[t + 1]
            
        # add (o, (x, y, z, wt), V(x)) to dataset
        for env_id in range(len(critics)):
            for t in range(n_timesteps - 1):
                o_ = obs[t, env_id]
                params = np.array([
                    env_params[env_id]['x'],
                    env_params[env_id]['y'],
                    env_params[env_id]['z'],
                    env_params[env_id]['wt']
                ])
                value = values[t, env_id]
                dataset.append((o_, params, value))

        pickle.dump(dataset, f)
