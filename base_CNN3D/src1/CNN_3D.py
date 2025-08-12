import time
import random
import os
import glob
import xarray as xr
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
from multiprocessing import Process, Manager, Value, Lock, Queue

import warnings
warnings.filterwarnings('ignore')




def open_datasets(args, train=True):
    """
    open datasets
    """
    pattern = os.path.join(args.data_dir, f"*{args.version}*.nc")
    all_files = sorted(glob.glob(pattern))

    if train:
        time_slice = args.train_time_slice
    else:
        time_slice = args.test_time_slice

    datasets = []
    for fn in all_files[time_slice]:

        ds_i = xr.open_dataset(fn, engine='netcdf4', chunks=args.chunks)
        datasets.append(ds_i)

    """
    concatenate along time dimension
    """
    combined = xr.concat(datasets, dim='time', combine_attrs='override')

    for ds in datasets:
        ds.close()

    return combined


def prepare_data_inds(ds, args, train=True):

    """
    choose far left upper corner of data cube without replacement
    """
    shape = ds[args.var].shape 
    shape_adj = (shape[0]-args.time_depth, shape[1]-args.y_height, shape[2]-args.x_width)

    # random indices for train and the same for test
    if train:
        random.seed(np.random.randint(1000))
        batches_per_epoch = args.train_batches_per_epoch

    else:
        random.seed(0)
        batches_per_epoch = args.test_batches_per_epoch

    # Convert 3D indices to flat indices
    total_points = np.prod(shape_adj, dtype=np.int64)

    # Choose N unique flat indices
    mask = ~np.isnan(ds[args.var][0].compute().values)
    total_data_cubes = int(1.1*batches_per_epoch*args.batch_size / (np.sum(mask) / mask.size))
    flat_indices = random.sample(range(total_points), total_data_cubes)

    # Convert flat indices back to 3D indices
    unique_indices = np.array(np.unravel_index(flat_indices, shape_adj)).T  # shape: (n_samples, 3)

    # take indices, which form will form a cube with more than "args.validity" fraction non nan values
    valid_unique_indices = []

    for ind in unique_indices:
        if np.sum(mask[ind[1]:ind[1]+args.y_height, ind[2]:ind[2]+args.x_width]) / args.y_height / args.x_width > args.validity:
            valid_unique_indices.append(ind)

    return np.array(valid_unique_indices)[:batches_per_epoch*args.batch_size]


def loader(ds, args, train=True):

    # prepare inds of data cubes
    inds = prepare_data_inds(ds, args, train=train)

    """
    create manager and shared variables for workers
    """
    finished_workers = 0
    manager = Manager()
    queue = manager.Queue(maxsize=args.batch_size*10)
    index_ptr = Value('i', 0)  # Shared integer index
    lock = Lock()
    stop_flags = [manager.Value('b', False) for _ in range(args.num_workers)]

    # Make indices sharable
    shared_ind = manager.list(inds.tolist())
    processes = []

    """
    start processes
    """
    for work_id in range(args.num_workers):
        
        p = Process(target=worker, args=(work_id, queue, lock, index_ptr, shared_ind, stop_flags, ds, args))
        p.start()
        processes.append(p)
        #print(f'process created {work_id}', flush=True)


    """
    main batch yielding loop
    """
    batch_x = []
    batch_y = []

    while True:

        cube = queue.get()

        if cube is None:

            finished_workers += 1
            if finished_workers == args.num_workers:

                '''
                x = torch.from_numpy(np.stack(batch_x)[:, None]).float() # [N,1,T,y,x]
                y = torch.from_numpy(np.stack(batch_y)).float() # [N,y,x]
                yield x, y 

                batch_x = []
                batch_y = []
                '''
                break

        else:

            batch_x.append(cube[:-1])
            batch_y.append(cube[-1])


        if len(batch_x) >= args.batch_size:

            #print(f'ind {index_ptr.value}, queue {queue.qsize()}', flush=True)
            x = torch.from_numpy(np.stack(batch_x)[:, None]).float() # [N,1,T,y,x]
            y = torch.from_numpy(np.stack(batch_y)).float() # [N,y,x]
            yield x, y 

            batch_x = []
            batch_y = []

    """
    close processes
    """
    for p in processes:
        p.join()


def worker(worker_id, queue, lock, index_ptr, shared_ind, stop_flags, ds, args):

    # perform cube extraction and push it into shared queue
    while True:

        with lock:

            if index_ptr.value >= len(shared_ind):
                stop_flags[worker_id].value = True
                break
            ind = shared_ind[index_ptr.value]
            index_ptr.value += 1

        queue.put(create_cube(ind, ds, args)) 

    queue.put(None)


def create_cube(ind, ds, args):

    # return one cube of data
    t = slice(ind[0], ind[0]+args.time_depth)
    y = slice(ind[1], ind[1]+args.y_height)
    x = slice(ind[2], ind[2]+args.x_width)

    return ds[args.var][t, y, x].fillna(0).compute().values



class Precip3DCNN(nn.Module):
    def __init__(self, input=1, hidden=32, time_depth=10):
        super().__init__()

        self.conv1 = nn.Conv3d(input, hidden, kernel_size=(3,3,3), padding=1)
        self.bn1 = nn.BatchNorm3d(hidden)
        self.conv2 = nn.Conv3d(hidden, hidden, kernel_size=(3,3,3), padding=1)
        self.bn2 = nn.BatchNorm3d(hidden)
        self.conv_time = nn.Conv3d(hidden, hidden, kernel_size=(time_depth,1,1))
        self.conv_out = nn.Conv2d(hidden, input, kernel_size=(3,3), padding=1)

    def forward(self, x):

        h = nn.functional.relu(self.bn1(self.conv1(x)))
        h = nn.functional.relu(self.bn2(self.conv2(h)))
        h = self.conv_time(h).squeeze(2)
        output = nn.functional.relu(self.conv_out(h))

        return output



def weighted_MSE(pred, true):

    true = true.flatten()
    pred = pred.flatten()

    weights = 1 / pd.Series(true).value_counts().sort_index()
    weights = torch.from_numpy(weights[true.numpy()].values) 

    return torch.mean(weights * (pred - true) ** 2)



def train(model, optimizer, criterion, train_loss, ds_train, args, start):
    
    # training phase
    model.train()

    work_done = 0
    print(f'train log: {work_done}/100, time: {time.time() - start}, loss: {np.mean(train_loss[-10:])}', flush=True)

    for x, y in loader(ds_train, args, train=True):

        x = x.to(args.device)
        y = y.to(args.device)

        optimizer.zero_grad()

        loss = criterion(model(x), y)
        loss.backward()

        optimizer.step()

        train_loss.append(loss.item())
        progress = len(train_loss) % args.train_batches_per_epoch / args.train_batches_per_epoch * 100

        if work_done <= progress // 1 :

            work_done += 1
            print(f'train log: {work_done}/100, time: {time.time() - start}, loss: {np.mean(train_loss[-10:])}', flush=True)
        

    return train_loss


def test(model, criterion, test_loss, ds_test, args, start):
        
    # test phase
    model.eval()

    loss_sum = 0
    print(f'test log: 0/1, time: {time.time() - start}, loss: nan', flush=True)

    with torch.no_grad():
        for x, y in loader(ds_test, args, train=False):

            x = x.to(args.device)
            y = y.to(args.device)

            loss = criterion(model(x), y)

            loss_sum += loss.item()

    test_loss.append(loss_sum / args.test_batches_per_epoch)

    print(f'test log: 1/1 time: {time.time() - start}, loss: {test_loss[-1]}', flush=True)

    return test_loss

    
def final_test(model, ds, args, exp_dir, start):

    # prepare model and other variables for ploting metric for the final test on the best checkpoint
    model.to(args.device)
    model.eval()
    preds = []
    trues = []
    
    # final test loop
    print(f'final test log: 0/1, time: {time.time() - start}', flush=True)
    with torch.no_grad():
        for x, y in loader(ds, args, train=False):
            
            x = x.to(args.device)

            pred = model(x).cpu().numpy()

            preds.append(pred)
            trues.append(y.numpy())


    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    test_dir = os.path.join(exp_dir, 'test')
    os.makedirs(test_dir, exist_ok=True)

    # evauate predicted values through histogram and scatter plot, and moment statistics
    flat_pred = preds.flatten()
    flat_true = trues.flatten()
    x = np.linspace(np.min(flat_true), np.max(flat_true), 200)
    plt.figure()
    plt.hist(flat_true, bins=x, density=True, log=True, alpha=0.5, label='true')
    plt.hist(flat_pred, bins=x, density=True, log=True, alpha=0.5, label='pred')
    plt.legend()
    plt.title('PDF Comparison')
    plt.savefig(os.path.join(test_dir, 'pdf_comparison.png'))
    plt.close()

    metrics = {
        'mean_true': flat_true.mean(),
        'mean_pred': flat_pred.mean(),
        'var_true': flat_true.var(),
        'var_pred': flat_pred.var(),
        'skew_true': stats.skew(flat_true),
        'skew_pred': stats.skew(flat_pred),
        'kurt_true': stats.kurtosis(flat_true),
        'kurt_pred': stats.kurtosis(flat_pred),
        'rmse': np.sqrt(np.mean((flat_pred-flat_true)**2))
    }
    pd.DataFrame(metrics, index=[0]).to_csv(os.path.join(test_dir, 'metrics.csv'), index=False)

    plt.figure()
    plt.scatter(flat_true, flat_pred, s=1, alpha=0.3)
    m = max(flat_true.max(), flat_pred.max())
    plt.plot([0,m],[0,m],'k--')
    plt.xlabel('True')
    plt.ylabel('Pred')
    plt.title('Scatter')
    plt.savefig(os.path.join(test_dir, 'scatter.png'))
    plt.close()

    print(f'final test log: 1/1, time: {time.time() - start}', flush=True)


def start_experiment(model, args, exp_dir):
    start = time.time()

    # open train and test datasets
    ds_train = open_datasets(args, train=True)
    ds_test = open_datasets(args, train=False)

    # initialze loss function and optimizer
    model.to(args.device)
    criterion = weighted_MSE
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    best_val = float('inf')

    # prepare path to directories
    checkp_dir = os.path.join(exp_dir, 'checkpoints')
    os.makedirs(checkp_dir, exist_ok=True)
    model_checkp = os.path.join(checkp_dir, 'model.pth')

    np.random.seed(0)
    torch.manual_seed(0)

    print(f'Total num. epochs: {args.epochs}', flush=True)
    print(f'Num. train batches per epoch: {args.train_batches_per_epoch}', flush=True)
    print(f'Num. test batches per epoch: {args.test_batches_per_epoch}', flush=True)
    
    # additional arrays for losses
    train_loss = []
    test_loss = []
    loss_png = os.path.join(exp_dir, 'loss_curve.png')

    # main train + test loop
    for epoch in range(1, args.epochs+1):

        train_loss = train(model, optimizer, criterion, train_loss, ds_train, args, start)

        test_loss = test(model, criterion, test_loss, ds_test, args, start)

        print(f"Epoch {epoch} finished, time: {time.time() - start}, train loss={np.mean(train_loss[-10:])}, test loss={test_loss[-1]}", flush=True)
        
        # update loss curve plot
        plt.figure()
        plt.semilogy(train_loss, label='train')
        plt.semilogy(np.linspace(1,len(train_loss),len(test_loss)), test_loss, label='test')
        plt.legend()
        plt.xlabel('Updates')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.savefig(loss_png)
        plt.close()

        
        # overwrite checkpoint on improvement
        if test_loss[-1] < best_val:

            best_val = test_loss[-1]
            torch.save(model.state_dict(), model_checkp)
            print(f"Saved model at epoch {epoch}, time {time.time() - start}", flush=True)


    # save final model
    final_checkp = os.path.join(checkp_dir, 'model_final.pth')
    torch.save(model.state_dict(), final_checkp)
    print(f"Saved final model", flush=True)


    # final test + metrics
    model.load_state_dict(torch.load(model_checkp))
    final_test(model, ds_test, args, exp_dir, start)


    print(f"finish", flush=True)

    




    

    