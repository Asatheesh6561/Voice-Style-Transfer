import os
import argparse
import tensorflow as tf
print(tf.__version__)
tf.config.optimizer.set_jit(True)

from solver import Solver
from data_loader import get_loader

def str2bool(v):
    return v.lower() in ('true')

def main(config):
    #set gpu memory allocation to be dynamic
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for i in physical_devices:
        tf.config.experimental.set_memory_growth(i, True)
    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)

    # Data loader.
    dataset = get_loader('train')
    
    solver = Solver(dataset, config)
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    solver.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Training configuration.
    parser.add_argument('--num_iters', type=int, default=100000, help='number of total iterations')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=2000, help='resume training from this step')

    # Miscellaneous.
    parser.add_argument('--use_tensorboard', type=str2bool, default=False)
    parser.add_argument('--device_id', type=int, default=0)

    # Directories.
    parser.add_argument('--log_dir', type=str, default='run/logs/')
    parser.add_argument('--model_save_dir', type=str, default='run/models/')
    parser.add_argument('--sample_dir', type=str, default='run/samples/')

    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=1000)

    config = parser.parse_args()
    print(config)

    main(config)