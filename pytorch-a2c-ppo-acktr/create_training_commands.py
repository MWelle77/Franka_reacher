import os
import argparse

import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run set of reacher experiments')
    parser.add_argument('save_directory', help='')
    parser.add_argument('--num-tasks', type=int, default=32,
                        help='number of tasks to train models for (default: 32)')
    parser.add_argument('--models-per-task', type=float, default=4,
                        help='for each task, train these many models (default: 4)')

    args = parser.parse_args()

    for task in range(args.num_tasks):
        wt = np.round(np.random.uniform(0.5, 1.0), 2)
        x = np.round(np.random.uniform(-0.1, 0.1), 2)
        y = np.round(np.random.uniform(-0.1, 0.1), 2)
        z = np.round(np.random.uniform(0.15, 0.2), 2)
        for model in range(args.models_per_task):
            model_id = 'reacher_{}_{}.pt'.format(task, model)
            print('python main.py {} --goal-x {} --goal-y {} --goal-z {} --euclidean-weight {} --num-processes 8 --num-frames 4000000'.format(
                os.path.join(args.save_directory, model_id),
                x,
                y,
                z,
                wt
            ))

