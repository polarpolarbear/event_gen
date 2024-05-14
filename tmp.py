import os
import argparse
import torch
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

  
cwd = os.getcwd()
print("Current Directory is: ", cwd)
prefix = os.path.abspath(os.path.join(cwd, os.pardir))  


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


print("END")
