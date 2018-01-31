import os
import argparse
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
from utils import *


def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training
    cudnn.benchmark = True

    # Create directories if not exist
    mkdir(config.log_path)
    mkdir(config.model_save_path)
    mkdir(config.sample_path)
    mkdir(config.result_path)

    data_loader = get_loader(config.data_path, config.image_size,
                             config.image_size, config.batch_size, 'PascalVOC2012', config.mode)

    # Solver
    solver = Solver(data_loader, vars(config))

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--image_size', type=int, default=96)
    parser.add_argument('--lr', type=float, default=0.0001)

    # Training settings

    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_epochs_decay', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--pretrained_model', type=str, default=None)

    # Misc
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=False)

    # Path
    parser.add_argument('--data_path', type=str, default='E:/Jonathan')
    parser.add_argument('--log_path', type=str, default='./lrnn/logs')
    parser.add_argument('--model_save_path', type=str, default='./lrnn/models')
    parser.add_argument('--sample_path', type=str, default='./lrnn/samples')
    parser.add_argument('--result_path', type=str, default='./lrnn/results')

    # Step size
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=100)
    parser.add_argument('--model_save_step', type=int, default=1000)

    config = parser.parse_args()

    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')

    main(config)