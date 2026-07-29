import time
import math
import random
import os
import glob
import xarray as xr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import torch.multiprocessing as mp

import warnings
warnings.filterwarnings('ignore')


def open_datasets(args):
    """
    open datasets
    """
    pattern = os.path.join(args.data_dir, f"*{args.version}*.nc")
    all_files = sorted(glob.glob(pattern))

    datasets = []
    for fn in all_files[args.time_slice]:

        ds_i = xr.open_dataset(fn, engine='netcdf4', chunks=args.chunks)
        datasets.append(ds_i)

    """
    concatenate along time dimension
    """
    combined = xr.concat(datasets, dim='time', combine_attrs='override')

    for ds in datasets:
        ds.close()

    return combined


def prepare_data_inds(ds, args, batches_per_epoch):

    # random indices for train and the same for test
    

    shape = ds[args.var].shape 
    shape_adj = (shape[1]-args.y_height, shape[2]-args.x_width)

    # Convert 2D indices to flat indices
    total_points = np.prod(shape_adj, dtype=np.int64)

    # Choose N unique flat indices
    mask = ~np.isnan(ds[args.var][0].compute().values)
    required_inds = int(batches_per_epoch * args.mini_batches_per_batch / args.mini_batch_per_coord)
    total_coordinates = int(required_inds * 1.1 / (np.sum(mask) / mask.size))
    flat_indices = random.sample(range(total_points), total_coordinates)

    # Convert flat indices back to 2D indices
    unique_indices = np.array(np.unravel_index(flat_indices, shape_adj)).T  # shape: (n_samples, 2)

    # take indices, which form will form a cube with more than "args.validity" fraction non nan values
    valid_unique_indices = []

    for ind in unique_indices:
        if np.sum(mask[ind[0] : ind[0] + args.y_height, ind[1] : ind[1] + args.x_width]) / args.y_height / args.x_width > args.validity:
            valid_unique_indices.append(ind)

    return np.array(valid_unique_indices)[ : required_inds]


def loader(ds, args, stage):

    if stage == 'train':
        batches_per_epoch = args.train_batches_per_epoch

    elif stage == 'test':
        batches_per_epoch = args.test_batches_per_epoch
    else:
        raise ValueError(stage)

    inds = prepare_data_inds(ds, args, batches_per_epoch).tolist()

    """
    create manager and shared variables for workers
    """
    index_ptr = mp.Value("i", 0)  
    lock = mp.Lock()
    queues = [
        mp.Queue(maxsize = args.mini_batches_per_batch // args.num_workers)
        for _ in range(args.num_workers)
    ]
    start_barrier = mp.Barrier(args.num_workers)
    shutdown_event = mp.Event()
    processes = []

    """
    start processes
    """
    for worker_id in range(args.num_workers):
        p = mp.Process(
            target=worker,
            args=(
                worker_id,
                queues[worker_id],
                lock,
                start_barrier,
                shutdown_event,
                index_ptr,
                inds,
                ds,
                args,
            ),
        )
        p.start()
        processes.append(p)
        #print(f'process created {work_id}', flush=True) 


    """
    main batch yielding loop
    """
    alive = [True] * args.num_workers
    batch = []

    while sum(alive) > 0:

        alive_ids = [i for i, a in enumerate(alive) if a]
        queue_ind = np.random.choice(alive_ids)

        mini_batch = queues[queue_ind].get()

        if mini_batch is None:
            alive[queue_ind] = False
            continue


        batch.append(mini_batch)


        if len(batch) >= args.mini_batches_per_batch:

            #print(f'ind {index_ptr.value}, queue {queue.qsize()}', flush=True)

            yield batch

            batch = []

    """
    close processes
    """
    shutdown_event.set()

    for p in processes:
        p.join()


def worker(worker_id,
            queue,
            lock,
            start_barrier,
            shutdown_event,
            index_ptr,
            inds,
            ds,
            args,):


    start_barrier.wait()

    # perform cube extraction and push it into shared queue
    while True:

        with lock:

            if index_ptr.value >= len(inds):
                break

            ind = inds[index_ptr.value]
            index_ptr.value += 1

            #print(f"worker: {worker_id}, index: {index_ptr.value}")

        for mini_batch in create_mini_batch(ind, ds, args):
            queue.put(mini_batch) 

    queue.put(None)

    shutdown_event.wait()


def create_mini_batch(ind, ds, args):

    data_pool = []

    t = slice(0, ds.time.size)
    y = slice(ind[0], ind[0] + args.y_height)
    x = slice(ind[1], ind[1] + args.x_width)

    column = ds['pr'][t, y, x].fillna(0).compute().values
    column_mean = column.mean()
    column_std = column.std()

    lat = ds['lat'][ind[0] + args.y_height // 2, ind[1] + args.x_width // 2].compute().values
    lat_mean = ds['lat'].mean().compute().values
    lat_std = ds['lat'].std().compute().values
    lat = (lat - lat_mean) / lat_std

    lon = ds['lon'][ind[0] + args.y_height // 2, ind[1] + args.x_width // 2].compute().values
    lon_mean = ds['lon'].mean().compute().values
    lon_std = ds['lon'].std().compute().values
    lon = (lon - lon_mean) / lon_std

    
    for _ in range(args.mini_batch_per_coord):

        mini_batch = column[np.random.choice(np.arange(ds.time.size), args.mini_batch_size)]

        if args.standardization:
            mini_batch = (mini_batch - column_mean) / column_std
        
        mini_batch = torch.from_numpy(mini_batch[:, None]).float()

        data_pool.append((mini_batch, 
                          torch.FloatTensor([lat, lon]),
                          (column_mean, column_std, column.max())))

    return data_pool



class FiLMBlock(nn.Module):
    def __init__(self, cond_dim, scale_factor, in_ch, out_ch, kernel_size, padding):
        super().__init__()

        self.upscale = nn.Upsample(scale_factor=scale_factor)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, padding_mode='replicate')
        self.film = nn.Linear(cond_dim, 2 * out_ch)

    def forward(self, x, cond):

        gamma, beta = self.film(cond).chunk(2, dim=-1)
        gamma = gamma[..., None, None]
        beta  = beta[..., None, None]

        return F.gelu(self.conv(self.upscale(x) * gamma + beta))



class PrecipCVAE(nn.Module):
    def __init__(self, out_ch, hid_ch, z_dim, cond_dim, device):
        super().__init__()

        self.z_dim = z_dim
        self.conditionals = cond_dim
        self.device = device
        
        self.init_latent_decoder = nn.Sequential(
            nn.Linear(z_dim, 400),
            nn.GELU(),
            nn.Linear(400, hid_ch*5*5),
            nn.GELU(),
            nn.Unflatten(1, (hid_ch, 5, 5))
        )
        self.cond_decoder = nn.Sequential(
            FiLMBlock(cond_dim, 2, hid_ch, hid_ch, kernel_size=(3,3), padding=1),
            FiLMBlock(cond_dim, 2, hid_ch, hid_ch, kernel_size=(3,3), padding=1),
            FiLMBlock(cond_dim, 1, hid_ch, hid_ch, kernel_size=(3,3), padding=1), 
        )
        self.output = nn.Conv2d(hid_ch, out_ch, kernel_size=(3,3), padding=1, padding_mode='replicate')

    def forward(self, batch_size, conds):
         
        e = (-torch.log(-torch.rand(batch_size, self.z_dim) + 1) - 1).to(self.device) 

        x = self.init_latent_decoder(e)
        #x = self.cond_latent_decoder(x, conds)
        #x = self.unflat(x)

        conds_expan = conds.expand(batch_size, -1)

        for block in self.cond_decoder:
            x = block(x, conds_expan)

        return self.output(x)



def moving_average(x, window):
    x = np.concatenate(([x[0]] * (window-1), x))
    return np.convolve(x, np.ones(window)/window, mode='valid')


def global_grad_norm(model):

    total_norm = 0

    for p in model.parameters():
        if p.grad is not None:
            total_norm += torch.sum(torch.pow(p.grad.data, 2)).item()

    total_norm = total_norm ** 0.5
        
    return total_norm


def ks_distance(a, b):

    data = np.sort(np.concatenate([a, b]))

    ecdf_a = np.searchsorted(np.sort(a), data, side='right') / len(a)
    ecdf_b = np.searchsorted(np.sort(b), data, side='right') / len(b)
    
    D = np.sort(np.unique(np.abs(ecdf_a - ecdf_b)))
    D_first = D[-1]
    D_second = D[-2]

    return D_first, D_second, data, ecdf_a, ecdf_b


def MMD(x, y, w, sigma=[0.01, 0.1, 1, 10, 100]):

    dim = x.shape[-1]


    dxx = torch.cdist(x, x, p=2) ** 2 / dim
    dyy = torch.cdist(y, y, p=2) ** 2 / dim


    xyw = x @ (y * w).T
    rxw = (x * x) @ w.T
    ryw = torch.sum(y * y * w, dim=1, keepdim=True).expand(-1, y.size(0)).T
    dxyw = torch.abs(ryw - 2. * xyw + rxw) / dim


    dxx = torch.nan_to_num(dxx)
    dyy = torch.nan_to_num(dyy)
    dxyw = torch.nan_to_num(dxyw)
    

    mmd = 0
    for s in sigma:

        K_xx = torch.exp(-dxx / s)
        K_yy = torch.exp(-dyy / s)
        K_xyw = torch.exp(-dxyw / s)

        mmd = mmd + K_xx.mean() + K_yy.mean() - 2 * K_xyw.mean()
        
    return mmd


def cvae_loss(x_hat, x, x_params, args):

    x = x.view(x.size(0), -1)
    x_hat = x_hat.view(x_hat.size(0), -1)

    x_og_mean, x_og_std, x_og_max = x_params
    w_min, w_max = args.dw_range

    if args.standardization:
        w = (x * x_og_std + x_og_mean) / x_og_max * (w_max - w_min) + w_min
    else:
        w = x / x_og_max * (w_max - w_min) + w_min

    recon_loss = MMD(x_hat, x, w)

    return recon_loss 


def get_linear_warmup_cosine_decay_scheduler(use, optimizer, total_steps, warmup, min_coef):
    
    warmup_steps = total_steps * warmup

    def lr_lambda(step):
        if use:
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            
            else:
                progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return 0.5 * (math.cos(math.pi * progress) * (1 - min_coef) + (1 + min_coef))
        
        else:
            return 1.0
        
    
    return LambdaLR(optimizer, lr_lambda)


def train(model, 
          optimizer,
          scheduler,
          criterion,
          train_things, 
          dataset, 
          args, 
          start):
    
    # training phase
    model.train()

    work_done = 0
    print(f"train log: {work_done}/100, time: {time.time() - start}, loss: {np.mean(train_things['loss'][-10:])}", flush=True)

    for train_data in loader(dataset, args, stage='train'):

        loss = 0

        for x, conds, x_params in train_data:

            x = x.to(args.device)
            conds = conds.to(args.device)

            optimizer.zero_grad()

            x_hat = model(args.mini_batch_size, conds)

            loss += criterion(x_hat, x, x_params, args)

        loss /= args.mini_batches_per_batch

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_clip)
        
        optimizer.step()
        scheduler.step()

        train_things['loss'].append(loss.item())
        train_things['grad'].append(global_grad_norm(model))

        progress = len(train_things['loss']) % args.train_batches_per_epoch / args.train_batches_per_epoch * 100

        if work_done <= progress // 1 :

            work_done += 1
            print(f"train log: {work_done}/100, time: {time.time() - start}, loss: {np.mean(train_things['loss'][-10:])}", flush=True)
        

    return train_things


def test(model, 
         criterion,
         test_things,
         dataset, 
         args, 
         start):
        
    # test phase
    model.eval()

    loss_sum = 0
    print(f"test log: 0/1, time: {time.time() - start}, loss: nan", flush=True)

    with torch.no_grad():
        for test_data in loader(dataset, args, stage='test'):

            loss = 0

            for x, conds, x_params in test_data:

                x = x.to(args.device)
                conds = conds.to(args.device)

                x_hat = model(args.mini_batch_size, conds)

                loss += criterion(x_hat, x, x_params, args)

            loss /= args.mini_batches_per_batch

            loss_sum += loss.item()

    test_things['loss'].append(loss_sum / args.test_batches_per_epoch)

    print(f"test log: 1/1 time: {time.time() - start}, loss: {test_things['loss'][-1]}", flush=True)

    return test_things

    
def final_test(model, 
               ds_test, 
               args, 
               exp_dir, 
               start):

    # prepare model and other variables for ploting metric for the final test on the best checkpoint
    model.to(args.device)
    model.eval()
    preds = []
    trues = []
    
    # final test loop
    print(f"final test log: 0/1, time: {time.time() - start}", flush=True)
    with torch.no_grad():
        for x in loader(ds_test, args, stage='final_test'):
            
            pred = model(args.batch_size)

            preds.append(pred.cpu().numpy())
            trues.append(x.numpy())


    trues = np.concatenate(trues, axis=0)
    preds = np.concatenate(preds, axis=0)

    test_dir = os.path.join(exp_dir, 'test')
    os.makedirs(test_dir, exist_ok=True)

    # evauate predicted values through histogram and scatter plot, and moment statistics
    flat_true = trues.flatten()
    flat_pred = preds.flatten()

    if args.standardization:
        flat_true = flat_true * data_std + data_mean
        flat_pred = flat_pred * data_std + data_mean

    fig = plt.figure(figsize=(10,30), layout='constrained')
    ax1 = fig.add_subplot(3,1,1)
    ax2 = fig.add_subplot(3,1,2)
    ax3 = fig.add_subplot(3,1,3)

    x = np.linspace(np.min(flat_true), np.max(flat_true), 200)
    ax1.hist(flat_true, bins=x, density=True, log=True, alpha=0.5, label=f'true\nmin = {flat_true.min()}\nmax = {flat_true.max()}')
    ax1.hist(flat_pred, bins=x, density=True, log=True, alpha=0.5, label=f'pred\nmin = {flat_pred.min()}\nmax = {flat_pred.max()}')
    ax1.set_title('PDF Comparison')
    ax1.set_ylabel('Density')
    ax1.legend()
    

    D_f, D_s, x, cdf_true, cdf_pred = ks_distance(flat_true, np.round(np.maximum(0, flat_pred), 1))
    ax2.plot(x, cdf_true, label=f'true')
    ax2.plot(x, cdf_pred, label=f'pred\nK_dist_f = {np.round(D_f*100, 2)}%\nK_dist_s = {np.round(D_s*100, 2)}%')
    ax2.set_ylabel('Cumulative Probability')
    ax2.set_title('Kolmogorov-Smirnov Test')
    ax2.legend()


    ax3.scatter(flat_true, flat_pred, s=1, alpha=0.3)
    m = max(flat_true.max(), flat_pred.max())
    ax3.plot([0,m],[0,m],'k--')
    ax3.set_xlabel('True')
    ax3.set_ylabel('Pred')
    ax3.set_title('Scatter')
    ax3.legend()


    fig.savefig(os.path.join(test_dir, 'pdf_comparison.png'))


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

    print(f"final test log: 1/1, time: {time.time() - start}", flush=True)


def start_experiment(model, 
                     args, 
                     exp_dir):
    
    start = time.time()
    print(f"Start experiment, time: {time.time() - start}", flush=True)

    # open train and test datasets
    dataset = open_datasets(args)

    # initialze loss function and optimizer
    model.to(args.device)
    criterion = cvae_loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = get_linear_warmup_cosine_decay_scheduler(args.use_scheduler,
                                                        optimizer,
                                                        total_steps = args.epochs*args.train_batches_per_epoch,
                                                        warmup = args.scheduler_warmup,
                                                        min_coef = args.min_scheduler_coef)
    best_val = float('inf')

    # prepare path to directories
    checkp_dir = os.path.join(exp_dir, 'checkpoints')
    os.makedirs(checkp_dir, exist_ok=True)
    model_checkp = os.path.join(checkp_dir, 'model.pth')


    print(f'Total num. epochs: {args.epochs}', flush=True)
    print(f'Num. train batches per epoch: {args.train_batches_per_epoch}', flush=True)
    print(f'Num. test batches per epoch: {args.test_batches_per_epoch}', flush=True)
    print(f'Num. final test batches per epoch: {args.final_test_batches_per_epoch}', flush=True)
    
    # additional arrays for losses
    train_things = {
        'loss': [],
        'grad': [],
    }
    test_things = {
        'loss': [],
    }
    loss_png = os.path.join(exp_dir, 'loss_curve.png')

    # main train + test loop
    for epoch in range(1, args.epochs+1):

        print(f"Epoch {epoch} started, time: {time.time() - start}", flush=True)

        train_things = train(model, 
                            optimizer,
                            scheduler,
                            criterion,
                            train_things,
                            dataset, 
                            args, 
                            start)

        test_things = test(model, 
                         criterion,
                         test_things,
                         dataset, 
                         args, 
                         start)

        print(f"Epoch {epoch} finished, time: {time.time() - start}, train loss={np.mean(train_things['loss'][-10:])}, test loss={test_things['loss'][-1]}", flush=True)
        
        # update loss curve plot
        fig = plt.figure(figsize=(20,20), layout='constrained')
        ax1 = fig.add_subplot(2,2,1)
        ax2 = fig.add_subplot(2,2,2)
        ax3 = fig.add_subplot(2,2,3)
        ax4 = fig.add_subplot(2,2,4)

        ax1.semilogy(train_things['loss'], label='train')
        ax1.semilogy(moving_average(train_things['loss'], 50), color='olive', label='train_ma')
        ax1.semilogy(np.linspace(1,len(train_things['loss']),len(test_things['loss'])), test_things['loss'], label='test')
        ax1.set_xlabel('Updates')
        ax1.set_ylabel('Loss')
        ax1.legend()

        ax2.semilogy(train_things['grad'], label='global_grad_norm')
        ax2.set_xlabel('Updates')
        ax2.set_ylabel('Grad_norm')
        ax2.legend()
        '''
        ax3.plot(train_things['mu'], label='average_mu')
        ax3.set_xlabel('Updates')
        ax3.set_ylabel('mu')
        ax3.legend()
        
        ax4.plot(np.exp(np.array(train_things['loglam'])), label='average_lam')
        ax4.set_xlabel('Updates')
        ax4.set_ylabel('lambda')
        ax4.legend()
        '''
        fig.savefig(loss_png)

        
        # overwrite checkpoint on improvement
        if test_things['loss'][-1] < best_val:

            best_val = test_things['loss'][-1]
            torch.save(model.state_dict(), model_checkp)
            print(f"Saved model at epoch {epoch}, time {time.time() - start}", flush=True)


    # save final model
    final_checkp = os.path.join(checkp_dir, 'model_final.pth')
    torch.save(model.state_dict(), final_checkp)
    print(f"Saved final model", flush=True)


    # final test + metrics
    '''
    model.load_state_dict(torch.load(model_checkp))
    print(f"Final test, time: {time.time() - start}", flush=True)
    final_test(model, 
               ds_test, 
               args, 
               exp_dir, 
               start)

    '''
    print(f"Finish experiment, time: {time.time() - start}", flush=True)

    




    

    