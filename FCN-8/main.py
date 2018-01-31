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
    data_loader = {}

    if config.dataset == 'Flowers':
        data_loader['Flowers'] = get_loader(config.flowers_image_path, config.flowers_crop_size,
                                 config.image_size, config.batch_size, 'Flowers', config.mode)

    # Solver
    solver = Solver(data_loader, vars(config))

    if config.mode == 'train':
        if config.dataset in ['CelebA', 'Flowers', 'RaFD']:
            solver.train()
    elif config.mode == 'test':
        if config.dataset in ['CelebA', 'Flowers', 'RaFD']:
            solver.test()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--c_dim', type=int, default=5)
    parser.add_argument('--i_dim', type=int, default=103)
    parser.add_argument('--c2_dim', type=int, default=8)
    parser.add_argument('--celebA_crop_size', type=int, default=178)
    parser.add_argument('--rafd_crop_size', type=int, default=256)
    parser.add_argument('--flowers_crop_size', type=int, default=100)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--g_conv_dim', type=int, default=64)#16)
    parser.add_argument('--d_conv_dim', type=int, default=64)
    parser.add_argument('--g_repeat_num', type=int, default=6)
    parser.add_argument('--d_repeat_num', type=int, default=6)
    parser.add_argument('--g_lr', type=float, default=0.0001)
    parser.add_argument('--d_lr', type=float, default=0.0001)
    parser.add_argument('--lambda_cls', type=float, default=1)
    parser.add_argument('--lambda_rec', type=float, default=10)
    parser.add_argument('--lambda_gp', type=float, default=10)
    parser.add_argument('--d_train_repeat', type=int, default=5)

    # Training settings
    parser.add_argument('--dataset', type=str, default='CelebA', choices=['CelebA','Flowers', 'RaFD', 'Both'])
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--num_epochs_decay', type=int, default=10)
    parser.add_argument('--num_iters', type=int, default=200000)
    parser.add_argument('--num_iters_decay', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--pretrained_model', type=str, default=None)

    # Test settings
    parser.add_argument('--test_model', type=str, default='20_1000')

    # Misc
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=False)

    # Path
    parser.add_argument('--celebA_image_path', type=str, default='./data/CelebA_nocrop/images')
    parser.add_argument('--rafd_image_path', type=str, default='./data/RaFD/train')
    parser.add_argument('--flowers_image_path', type=str, default='../StarGAN/7386_flowers.hdf5')
    parser.add_argument('--metadata_path', type=str, default='./data/list_attr_celeba.txt')
    parser.add_argument('--log_path', type=str, default='./stargan/logs')
    parser.add_argument('--model_save_path', type=str, default='./stargan/models')
    parser.add_argument('--sample_path', type=str, default='./stargan/samples')
    parser.add_argument('--result_path', type=str, default='./stargan/results')

    # Step size
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=500)
    parser.add_argument('--model_save_step', type=int, default=1000)

    config = parser.parse_args()

    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')

    main(config)