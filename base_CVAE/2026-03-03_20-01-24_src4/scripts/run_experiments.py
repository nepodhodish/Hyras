# nohup python3 run_experiment.py 2>&1 &
# disown
import time
import random
import os
import sys
import glob
import shutil
import argparse
import datetime
import numpy as np
import torch
import warnings
import torch.multiprocessing as mp

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
MAIN_DIR = os.path.dirname(PROJECT_DIR)
import CVAE 




def main():

    # parameters
    parser = argparse.ArgumentParser(description='Run 3D-CNN precipitation experiments')
    parser.add_argument('--data_dir', type=str, default=os.path.join(MAIN_DIR, 'data1'), 
                        help='Path to data folder')
    parser.add_argument('--version', type=str, default='v6-1', 
                        help='Data version to use')
    parser.add_argument('--chunks', type=dict, default={'time': 100}, 
                        help='Chunking rule for loading data')
    parser.add_argument('--time_slice', type=slice, default=slice(0, 96), 
                        help='Which years to take for training')
    parser.add_argument('--epochs', type=int, default=20, 
                        help='Number of training epochs')
    parser.add_argument('--train_batches_per_epoch', type=int, default=500, 
                        help='Num. train batches per epoch')
    parser.add_argument('--test_batches_per_epoch', type=int, default=50, 
                        help='Num. test batches per epoch')
    parser.add_argument('--final_test_batches_per_epoch', type=int, default=25, 
                        help='Num. final test batches per epoch')
    parser.add_argument('--mini_batch_size', type=int, default=500, 
                        help='Training batch size')
    parser.add_argument('--mini_batches_per_batch', type=int, default=15, 
                        help='Num. conditional mini_batches in batch')
    parser.add_argument('--mini_batch_per_coord', type=int, default=5, 
                        help='Num. mini-batches per coordinate')
    parser.add_argument('--standardization', type=bool, default=True, 
                        help='Data standardization')
    parser.add_argument('--hidden_ch', type=int, default=96, 
                        help='Num. model hidden channels')
    parser.add_argument('--z_dim', type=int, default=2, 
                        help='Dimension model latent layer')
    parser.add_argument('--cond_dim', type=int, default=2, 
                        help='Num. conditional variables')
    parser.add_argument('--var', type=str, default='pr', 
                        help='We are working with precipitation data')
    parser.add_argument('--validity', type=float, default=0.9, 
                        help='Minimum fraction of valid non nan values inside data cube')
    parser.add_argument('--num_workers', type=int, default=30, 
                        help='Number workers for dataloader')
    parser.add_argument('--y_height', type=int, default=20, 
                        help='Height of data cube along Y axis')
    parser.add_argument('--x_width', type=int, default=20, 
                        help='Width of data cube along X axis')
    parser.add_argument('--use_scheduler', type=bool, default=True, 
                        help='Use scheduler or not') 
    parser.add_argument('--scheduler_warmup', type=float, default=0.0, 
                        help='Linear warmup for scheduler') 
    parser.add_argument('--min_scheduler_coef', type=float, default=1e-1, 
                        help='Minimum multiplicator scheduler achieves') 
    parser.add_argument('--lr', type=float, default=1e-3, 
                        help='Learning rate')
    parser.add_argument('--wd', type=float, default=1e-3, 
                        help='Weight decay')
    parser.add_argument('--dw_range', type=tuple, default=(1, 1.5), 
                        help='Data weight range')
    parser.add_argument('--norm_clip', type=float, default=50.0, 
                        help='Norm clip threshold')
    parser.add_argument('--device', type=str, default=('cuda:4' if torch.cuda.is_available() else 'cpu'), help='Device')
    args = parser.parse_args()


    # create experiment folder
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    exp_dir = os.path.join(PROJECT_DIR, 'experiments', f'{timestamp}_{os.path.basename(SCRIPT_DIR)}')
    os.makedirs(exp_dir, exist_ok=True)


    # establish main model
    model = CVAE.PrecipCVAE(out_ch = 1, 
                          hid_ch = args.hidden_ch, 
                          z_dim = args.z_dim,
                          cond_dim = args.cond_dim,
                          device = args.device)

    
    # count number of parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    parser.add_argument('--total_parameters', type=int, default=total_params, 
                        help='Total num. parameters')


    # save settings
    args = parser.parse_args()
    settings_path = os.path.join(exp_dir, 'experiment_settings.txt')
    with open(settings_path, 'w') as file:
        file.write(f'Command: {" ".join(sys.argv)} \n')
        file.write(f'Arguments:\n')
        for k,v in vars(args).items():
            file.write(f'  {k}: {v}\n')


    # backup scripts
    scripts_dir = os.path.join(exp_dir, 'scripts')
    os.makedirs(scripts_dir, exist_ok=True)
    for file in glob.glob(os.path.join(SCRIPT_DIR, '*.py')):
        shutil.copy(file, scripts_dir)

    

    # run training & test
    CVAE.start_experiment(model, args, exp_dir)










if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    main()
