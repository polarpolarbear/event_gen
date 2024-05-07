import numpy as np
import os
import random
from tqdm import tqdm
from geomloss import SamplesLoss  # See also ImagesLoss, VolumesLoss
import cv2
from dv import AedatFile
import pickle
from event_reading import read_event
from tonic.io import read_mnist_file
   
cwd = os.getcwd()
print("Current Directory is: ", cwd)
prefix = os.path.abspath(os.path.join(cwd, os.pardir))  
N_MNIST_dir = prefix +"/data/Caltech101DVS_jelly/events_np"
random.seed(42)

class_path_list = os.listdir(N_MNIST_dir)
max_x_list = []
max_y_list = []
nEvent_list = []
label_list = []
for iClass,class_path in enumerate(class_path_list):
    max_x_list.append([])
    max_y_list.append([])
    nEvent_list.append([])
    label_list.append(f"{iClass}:{class_path}")
    print(f"{iClass}:{class_path}")
    N_MNIST_class_path = os.path.join(N_MNIST_dir, class_path)
    file_path_list = os.listdir(N_MNIST_class_path)
    for file_path in file_path_list:        
        N_MNIST_file_path = os.path.join(N_MNIST_class_path, file_path)
        #xytp = read_mnist_file(N_MNIST_file_path,np.dtype([("x", int), ("y", int), ("t", int), ("p", int)]))
        xytp = np.load(N_MNIST_file_path)
        mask = np.logical_and(200000-100000/8 < xytp['t'], xytp['t'] < 200000)
        x = xytp['x'][mask]
        y = xytp['y'][mask]
        t = xytp['t'][mask]
        p = xytp['p'][mask]
        nEvent_list[-1].append(len(x))
        max_x_list[-1].append(np.max(x))
        max_y_list[-1].append(np.max(y))

with open('caltech_new_nEvent.pickle', 'wb') as handle:
    pickle.dump(nEvent_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('label.pickle', 'wb') as handle:
    pickle.dump(label_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('caltech_max_x.pickle', 'wb') as handle:
    pickle.dump(max_x_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('caltech_max_y.pickle', 'wb') as handle:
    pickle.dump(max_y_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
