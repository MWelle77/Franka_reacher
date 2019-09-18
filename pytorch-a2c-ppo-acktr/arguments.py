import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('save_path', help='filename to save trained model as')
    parser.add_argument('--algo', default='ppo',
                        help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='learning rate (default: 3e-4)')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--use-gae', action='store_true', default=True,
                        help='use generalized advantage estimation')
    parser.add_argument('--tau', type=float, default=0.95,
                        help='gae parameter (default: 0.95)')
    parser.add_argument('--entropy-coef', type=float, default=0.0,
                        help='entropy term coefficient (default: 0.0)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--num-processes', type=int, default=2,
                        help='how many training CPU processes to use (default: 2)')
    parser.add_argument('--num-steps', type=int, default=2048,
                        help='number of roll-out steps (default: 2048)')
    parser.add_argument('--ppo-epoch', type=int, default=10,
                        help='number of ppo epochs (default: 10)')
    parser.add_argument('--num-mini-batch', type=int, default=32,
                        help='number of batches for ppo (default: 32)')
    parser.add_argument('--clip-param', type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument('--num-stack', type=int, default=1,
                        help='number of frames to stack (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1,
                        help='log interval, one log per n updates (default: 1)')
    parser.add_argument('--vis-interval', type=int, default=1,
                        help='vis interval, one log per n updates (default: 1)')
    parser.add_argument('--num-frames', type=int, default=1e6,
                        help='number of frames to train (default: 1e6)')
    parser.add_argument('--env-name', default='YumiReacher-v0',
                        help='environment to train on (default: YumiReacher-v0)')
    parser.add_argument('--log-dir', default='/tmp/gym/',
                        help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--recurrent-policy', action='store_true', default=False,
                        help='use a recurrent policy')
    parser.add_argument('--no-vis', action='store_true', default=False,
                        help='disables visdom visualization')
    parser.add_argument('--port', type=int, default=8097,
                        help='port to run the server on (default: 8097)')
    parser.add_argument('--euclidean-weight', type=float, default=0.9,
                        help='weight on the euclidean part of the reward (default: 0.9)')
    parser.add_argument('--goal-x', type=float, default=0.0,
                        help='goal x-position (default: 0.0)')
    parser.add_argument('--goal-y', type=float, default=0.0,
                        help='goal y-position (default: 0.0)')
    parser.add_argument('--goal-z', type=float, default=0.2,
                        help='goal z-position (default: 0.2)')
    parser.add_argument('--random-task', action='store_true', default=False,
                        help='randomly assign wt, x, y, z')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.vis = not args.no_vis

    return args
