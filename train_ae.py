import os
import argparse
import torch
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from utils.dataset import *
from utils.misc import *
from utils.data import *
from utils.transform import *
from models.autoencoder import *
from evaluation import EMD_CD

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from fc_ae import PointNetEncoder, PointNetDecoder,imageEncoder,BasicBlock
from event_reading import read_event
import torchvision.transforms as transforms
import os
import random
from tqdm import tqdm
from geomloss import SamplesLoss  # See also ImagesLoss, VolumesLoss
import cv2
#from apex import amp
from sklearn.metrics import r2_score

torch.cuda.empty_cache()


# Arguments
parser = argparse.ArgumentParser()
# Model arguments
parser.add_argument('--latent_dim', type=int, default=256)
parser.add_argument('--num_steps', type=int, default=200)
parser.add_argument('--beta_1', type=float, default=1e-4)
parser.add_argument('--beta_T', type=float, default=0.05)
parser.add_argument('--sched_mode', type=str, default='linear')
parser.add_argument('--flexibility', type=float, default=0.0)
parser.add_argument('--residual', type=eval, default=True, choices=[True, False])
parser.add_argument('--resume', type=str, default=None)

# Datasets and loaders
parser.add_argument('--dataset_path', type=str, default='./data/shapenet.hdf5')
parser.add_argument('--categories', type=str_list, default=['airplane'])
parser.add_argument('--scale_mode', type=str, default='shape_unit')
parser.add_argument('--train_batch_size', type=int, default=128)
parser.add_argument('--val_batch_size', type=int, default=32)
parser.add_argument('--rotate', type=eval, default=False, choices=[True, False])

# Optimizer and scheduler
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--max_grad_norm', type=float, default=10)
parser.add_argument('--end_lr', type=float, default=1e-4)
parser.add_argument('--sched_start_epoch', type=int, default=150*THOUSAND)
parser.add_argument('--sched_end_epoch', type=int, default=300*THOUSAND)

# Training
parser.add_argument('--seed', type=int, default=2020)
parser.add_argument('--logging', type=eval, default=True, choices=[True, False])
parser.add_argument('--log_root', type=str, default='./logs_ae')
parser.add_argument('--device', type=str, default='cuda:3')
parser.add_argument('--max_iters', type=int, default=float('inf'))
parser.add_argument('--val_freq', type=float, default=1000)
parser.add_argument('--tag', type=str, default=None)
parser.add_argument('--num_val_batches', type=int, default=-1)
parser.add_argument('--num_inspect_batches', type=int, default=1)
parser.add_argument('--num_inspect_pointclouds', type=int, default=4)
args = parser.parse_args()
seed_all(args.seed)


def modify_file_name(file_path):
    nr = int(file_path.split('.')[0])
    new_path = str(nr-1)+".jpg"
    return new_path

def random_pad_with_c(N_MNIST,max_n_events):
  n_event = N_MNIST.shape[0]
  padded_events = np.zeros((max_n_events, 5))  
  padded_events[:n_event,:4] = N_MNIST  
    
  indices = np.random.choice(n_event, max_n_events - n_event, replace=True)  
  padded_events[n_event:,:4] = N_MNIST[indices]


  padded_events[:n_event,4] = 1
  #padded_events[n_event:,4] = np.random.normal(0.25, 0.5, max_n_events-n_event)
  #padded_events[:n_event,4] = np.random.normal(0.75, 0.5, n_event)
  return padded_events

def create_loader(N_MNIST_dir, MNIST_dir,seed,batchsize, max_n_events,split,save_stat):
  grayscale_transform = transforms.Grayscale()
  class_path_list = os.listdir(N_MNIST_dir)
  N_MNIST_list = []
  MNIST_list = []
  inputmap_list = []
  label_list = []

  nEvent_list = []
  max_time_list = []
  for class_path in class_path_list:
      nEvent_list.append([])
      max_time_list.append([])
      N_MNIST_class_path = os.path.join(N_MNIST_dir, class_path)
      MNIST_class_path = os.path.join(MNIST_dir, class_path)
      file_path_list = os.listdir(N_MNIST_class_path)
      for file_path in file_path_list:        
        N_MNIST_file_path = os.path.join(N_MNIST_class_path, file_path)
        N_MNIST = read_event(N_MNIST_file_path)
        ori_length = N_MNIST.shape[0]
        if len(nEvent_list[-1])<5400:
          nEvent_list[-1].append(ori_length)
          max_time_list[-1].append(N_MNIST[-1,2])        

        if ori_length > max_n_events:
          print(N_MNIST_file_path)
          continue
        
        N_MNIST = random_pad_with_c(N_MNIST,max_n_events)
        N_MNIST = torch.tensor(N_MNIST).to(torch.float)
        N_MNIST_list.append(N_MNIST)

        MNIST_file_path = os.path.join(MNIST_class_path, modify_file_name(file_path))
        MNIST = cv2.imread(MNIST_file_path)
        MNIST = cv2.resize(MNIST, (34, 34), interpolation=cv2.INTER_LINEAR)/255 
        MNIST = grayscale_transform(torch.tensor(MNIST).permute(2,0,1)).to(torch.float)
        MNIST_list.append(MNIST)

        #event_histogram_data = torch.tensor(event2histogram_alt(N_MNIST))
        #input_map = torch.cat((MNIST, event_histogram_data), dim=0).to(torch.float32)
        #inputmap_list.append(input_map)

        inputmap_list.append(1)
        pad = int(ori_length < max_n_events)
        label_list.append(torch.tensor([int(class_path),pad,ori_length]))

  if save_stat == True:
    np.save('nEvent_list.npy', nEvent_list)
    np.save('max_time_list.npy', max_time_list)

  merged_data = list(zip(N_MNIST_list, MNIST_list, inputmap_list, label_list))
  random.shuffle(merged_data)

  if split == False:
    data_loader = torch.utils.data.DataLoader(
        dataset=merged_data,
        batch_size=batchsize,
        shuffle=False,
        drop_last=True,
        pin_memory=True
    )
    return data_loader
  
  else:
    ii = int(0.7*len(merged_data))
    train_loader = torch.utils.data.DataLoader(
        dataset=merged_data[:ii],
        batch_size=batchsize,
        shuffle=False,
        drop_last=True,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset= merged_data[ii:],
        batch_size=batchsize,
        shuffle=True,
        drop_last=True,
        pin_memory=True
    )
    return train_loader,test_loader
  
# Logging
if args.logging:
    log_dir = get_new_log_dir(args.log_root, prefix='AE_', postfix='_' + args.tag if args.tag is not None else '')
    logger = get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    ckpt_mgr = CheckpointManager(log_dir)
else:
    logger = get_logger('train', None)
    writer = BlackHole()
    ckpt_mgr = BlackHole()
logger.info(args)

# Datasets and loaders
# transform = None
# if args.rotate:
#     transform = RandomRotate(180, ['pointcloud'], axis=1)
# logger.info('Transform: %s' % repr(transform))
# logger.info('Loading datasets...')
# train_dset = ShapeNetCore(
#     path=args.dataset_path,
#     cates=args.categories,
#     split='train',
#     scale_mode=args.scale_mode,
#     transform=transform,
# )
# val_dset = ShapeNetCore(
#     path=args.dataset_path,
#     cates=args.categories,
#     split='val',
#     scale_mode=args.scale_mode,
#     transform=transform,
# )
# train_iter = get_data_iterator(DataLoader(
#     train_dset,
#     batch_size=args.train_batch_size,
#     num_workers=0,
# ))
# val_loader = DataLoader(val_dset, batch_size=args.val_batch_size, num_workers=0)

cwd = os.getcwd()
print("Current Directory is: ", cwd)
prefix = os.path.abspath(os.path.join(cwd, os.pardir))  

#------------load data------------
N_MNIST_train_path = prefix +"/data/NMNIST_Train"
MNIST_train_path = prefix +"/data/MNIST_Train"
N_MNIST_test_path = prefix +"/data/NMNIST_Test"
MNIST_test_path = prefix +"/data/MNIST_Test"
batchsize = 32
max_n_events = 3324
seed = 42
train_data_loader = create_loader(N_MNIST_train_path,MNIST_train_path,seed,batchsize,max_n_events,split=False,save_stat=True)
test_data_loader = create_loader(N_MNIST_test_path,MNIST_test_path,seed,batchsize,max_n_events,split=False,save_stat=False)
print(f"train_data_loader_size: {len(train_data_loader)*batchsize}")
print(f"test_data_loader_size: {len(test_data_loader)*batchsize}")
print(f"max_n_events: {max_n_events}") 


# Model
logger.info('Building model...')
if args.resume is not None:
    logger.info('Resuming from checkpoint...')
    ckpt = torch.load(args.resume)
    model = AutoEncoder(ckpt['args']).to(args.device)
    model.load_state_dict(ckpt['state_dict'])
else:
    model = AutoEncoder(args).to(args.device)
logger.info(repr(model))


# Optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), 
    lr=args.lr, 
    weight_decay=args.weight_decay
)
scheduler = get_linear_scheduler(
    optimizer,
    start_epoch=args.sched_start_epoch,
    end_epoch=args.sched_end_epoch,
    start_lr=args.lr,
    end_lr=args.end_lr
)

# Train, validate 
def train(it):
    # Load data
    batch = next(train_iter)
    x = batch['pointcloud'].to(args.device)

    # Reset grad and model state
    optimizer.zero_grad()
    model.train()

    # Forward
    loss = model.get_loss(x)

    # Backward and optimize
    loss.backward()
    orig_grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()
    scheduler.step()

    logger.info('[Train] Iter %04d | Loss %.6f | Grad %.4f ' % (it, loss.item(), orig_grad_norm))
    writer.add_scalar('train/loss', loss, it)
    writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
    writer.add_scalar('train/grad_norm', orig_grad_norm, it)
    writer.flush()

def validate_loss(it):

    all_refs = []
    all_recons = []
    for i, batch in enumerate(tqdm(val_loader, desc='Validate')):
        if args.num_val_batches > 0 and i >= args.num_val_batches:
            break
        ref = batch['pointcloud'].to(args.device)
        shift = batch['shift'].to(args.device)
        scale = batch['scale'].to(args.device)
        with torch.no_grad():
            model.eval()
            code = model.encode(ref)
            recons = model.decode(code, ref.size(1), flexibility=args.flexibility)
        all_refs.append(ref * scale + shift)
        all_recons.append(recons * scale + shift)

    all_refs = torch.cat(all_refs, dim=0)
    all_recons = torch.cat(all_recons, dim=0)
    metrics = EMD_CD(all_recons, all_refs, batch_size=args.val_batch_size)
    cd, emd = metrics['MMD-CD'].item(), metrics['MMD-EMD'].item()
    
    logger.info('[Val] Iter %04d | CD %.6f | EMD %.6f  ' % (it, cd, emd))
    writer.add_scalar('val/cd', cd, it)
    writer.add_scalar('val/emd', emd, it)
    writer.flush()

    return cd

def validate_inspect(it):
    sum_n = 0
    sum_chamfer = 0
    for i, batch in enumerate(tqdm(val_loader, desc='Inspect')):
        x = batch['pointcloud'].to(args.device)
        model.eval()
        code = model.encode(x)
        recons = model.decode(code, x.size(1), flexibility=args.flexibility).detach()

        sum_n += x.size(0)
        if i >= args.num_inspect_batches:
            break   # Inspect only 5 batch

    writer.add_mesh('val/pointcloud', recons[:args.num_inspect_pointclouds], global_step=it)
    writer.flush()

# Main loop
logger.info('Start training...')
try:
    it = 1
    while it <= args.max_iters:
        train(it)
        if it % args.val_freq == 0 or it == args.max_iters:
            with torch.no_grad():
                cd_loss = validate_loss(it)
                validate_inspect(it)
            opt_states = {
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            ckpt_mgr.save(model, args, cd_loss, opt_states, step=it)
        it += 1

except KeyboardInterrupt:
    logger.info('Terminating...')
