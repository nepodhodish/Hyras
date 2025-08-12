import time
import os
import sys
import glob
import shutil
import argparse
import datetime
import numpy as np
import torch
import warnings
import multiprocessing

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
MAIN_DIR = os.path.dirname(PROJECT_DIR)
import CNN_3D




def main():


    # parameters
    parser = argparse.ArgumentParser(description='Run 3D-CNN precipitation experiments')
    parser.add_argument('--data_dir', type=str, default=os.path.join(MAIN_DIR, 'data1'), help='Path to data folder')
    parser.add_argument('--version', type=str, default='v6-1', help='Data version to use')
    parser.add_argument('--chunks', type=dict, default={'time': 100}, help='Chunking rule for loading data')
    parser.add_argument('--time_slice', type=slice, default=slice(90,91), help='Which years to take')
    parser.add_argument('--var', type=str, default='pr', help='We are working with precipitation data')
    
    parser.add_argument('--data_scale', type=float, default=1, help='Independent parameter for scaling train data amount')
    parser.add_argument('--validity', type=float, default=0.9, help='Minimum fraction of valid non nan values inside data cube')
    parser.add_argument('--test_split', type=float, default=0.2, help='Proportion of test data')
    parser.add_argument('--num_workers', type=int, default=20, help='Number workers for dataloader')

    parser.add_argument('--time_depth', type=int, default=10+1, help='Number of past timesteps for train+test')
    parser.add_argument('--y_height', type=int, default=20, help='Height of data cube along Y axis')
    parser.add_argument('--x_width', type=int, default=20, help='Width of data cube along X axis')
    parser.add_argument('--batch_size', type=int, default=100, help='Training batch size')

    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--wd', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--device', type=str, default=('cpu' if torch.cuda.is_available() else 'cpu'), help='Device')
    args = parser.parse_args()


    # create experiment folder
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    exp_dir = os.path.join(PROJECT_DIR, 'experiments', f'{timestamp}_{os.path.basename(SCRIPT_DIR)}')
    os.makedirs(exp_dir, exist_ok=True)


    # save settings
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

    
    # open main dataset
    ds = CNN_3D.open_datasets(args)
    # prepare train and test inds
    train_ind, test_ind = CNN_3D.prepare_data_inds(ds, args)

    print(f'Total num. epochs: {args.epochs}', flush=True)
    print(f'Num. train batches per epoch: {len(train_ind) // args.batch_size}', flush=True)
    print(f'Num. test batches per epoch: {len(test_ind) // args.batch_size}', flush=True)
    


    # build model and show params
    model = CNN_3D.Precip3DCNN(input=1, hidden=32, time_depth=args.time_depth-1)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total num. parameters: {total_params}', flush=True)


    # run training & test
    best_checkp = CNN_3D.train(model, train_ind, test_ind, ds, args, exp_dir)

    model.load_state_dict(torch.load(best_checkp))

    CNN_3D.test(model, test_ind, ds, args, exp_dir)









if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main()
