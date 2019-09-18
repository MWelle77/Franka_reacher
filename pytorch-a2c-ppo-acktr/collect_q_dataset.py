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

with open('q_dataset.pkl', 'wb') as f:
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
            mask = np.random.binomial(1, 0.8, size=(cpu_actions.shape[0], 1)).repeat(cpu_actions.shape[1], axis=1)
            cpu_actions = mask * cpu_actions + (1 - mask) * np.random.uniform(-2, 2, size=cpu_actions.shape)
            o, r, d, i = envs.step(cpu_actions)
            obs[t + 1] = o
            rewards[t, :, 0] = r
            actions[t] = cpu_actions
            
        # add (o, o', (x, y, z, wt), r) to dataset
        for env_id in range(len(critics)):
            for t in range(n_timesteps - 1):
                o = obs[t, env_id]
                o_ = obs[t + 1, env_id]
                params = np.array([
                    env_params[env_id]['x'],
                    env_params[env_id]['y'],
                    env_params[env_id]['z'],
                    env_params[env_id]['wt']
                ])
                dataset.append((o, o_, params, actions[t, env_id], rewards[t, env_id]))

        pickle.dump(dataset, f)
