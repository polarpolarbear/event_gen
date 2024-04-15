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

def chamfer_distance(y_pred, y):
    # y_pred: (B, N, D) tensor
    # y: (B, M, D) tensor

    batch_size = y_pred.size(0)
    #num_points_y_pred = y_pred.size(1)
    #num_points_y = y.size(1)
    #dim = y_pred.size(2)

    # Expand y_pred and y to have the same number of dimensions

    y_pred = y_pred[:,:,:3].unsqueeze(2)  # (B, N, 1, D)
    y = y[:,:,:3].unsqueeze(1)  # (B, 1, M, D)

    # Compute Euclidean distance
    dist_matrix = torch.sum((y_pred - y)**2, dim=-1)  # (B, N, M)

    # Find the nearest neighbor in y for each point in y_pred
    min_dist_y_pred_to_y, _ = torch.min(dist_matrix, dim=2)  # (B, N)

    # Find the nearest neighbor in x for each point in y
    min_dist_y_to_y_pred, _ = torch.min(dist_matrix, dim=1)  # (B, M)

    # Sum or Average (torch sum or torch mean) over all points
    chamfer_loss = torch.sum(min_dist_y_pred_to_y, dim=1) + torch.sum(min_dist_y_to_y_pred, dim=1)

    # Average over batch and number of points
    chamfer_loss = torch.mean(chamfer_loss)
    return chamfer_loss

def modify_file_name(file_path):
    nr = int(file_path.split('.')[0])
    new_path = str(nr-1)+".jpg"
    return new_path


def event2histogram_tri(event_stream):
    event_stream = event_stream.numpy()
    hist = np.zeros((34, 34, 3))
    xx = np.floor((event_stream[:,0]*33)).astype(int)
    yy= np.floor((event_stream[:,1]*33)).astype(int)
    pp = np.floor((event_stream[:,3])).astype(int)
    for p in range(3):
      tmp = np.zeros((34, 34))
      ii = np.where(pp == p)
      np.add.at(tmp, (yy[ii], xx[ii]), 1)
      hist[:,:,p] = tmp
    return hist

def event2histogram_mono(event_stream):
    event_stream = event_stream.numpy()
    hist = np.zeros((34, 34))
    xx = np.floor((event_stream[:,0]*33)).astype(int)
    yy= np.floor((event_stream[:,1]*33)).astype(int)
    np.add.at(hist, (yy, xx), 1)
    return hist

def random_pad_with_c(N_MNIST,max_n_events):
  n_event = N_MNIST.shape[0]
  padded_events = np.zeros((max_n_events, 5))  
  padded_events[:n_event,:4] = N_MNIST  
    
  indices = np.random.choice(n_event, max_n_events - n_event, replace=True)  
  padded_events[n_event:,:4] = N_MNIST[indices]
  padded_events[n_event:,4] = 1
  return padded_events

def random_pad_N_MIST(N_MNIST,max_n_events):
  n_event, _ = N_MNIST.shape
  padded_events = np.zeros((max_n_events, 4))  
  padded_events[:n_event] = N_MNIST  

  indices = np.random.choice(n_event, max_n_events - n_event, replace=True)  
  padded_events[n_event:] = N_MNIST[indices]
  padded_events[n_event:,3] = 0.5 # {0,1,2}
  return padded_events

def create_loader(N_MNIST_dir, MNIST_dir,seed,batchsize, max_n_events,split):
  if seed != 0:
    random.seed(seed)
  grayscale_transform = transforms.Grayscale()
  class_path_list = os.listdir(N_MNIST_dir)
  N_MNIST_list = []
  MNIST_list = []
  inputmap_list = []
  label_list = []

  nEvent_list = []
  for class_path in class_path_list:
      N_MNIST_class_path = os.path.join(N_MNIST_dir, class_path)
      MNIST_class_path = os.path.join(MNIST_dir, class_path)
      file_path_list = os.listdir(N_MNIST_class_path)
      for file_path in file_path_list:        
        N_MNIST_file_path = os.path.join(N_MNIST_class_path, file_path)
        N_MNIST = read_event(N_MNIST_file_path)
        ori_length = N_MNIST.shape[0]
        nEvent_list.append(ori_length)
        
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

  np.save('nEvent_list.npy', nEvent_list)
  #np.save('N_MNIST.npy', N_MNIST_list[0])
  merged_data = list(zip(N_MNIST_list, MNIST_list, inputmap_list, label_list))
  random.shuffle(merged_data)

  if split == False:
    data_loader = torch.utils.data.DataLoader(
        dataset=merged_data,
        batch_size=batchsize,
        shuffle=True,
        drop_last=True,
        pin_memory=True
    )
    return data_loader
  
  else:
    ii = int(0.7*len(merged_data))
    train_loader = torch.utils.data.DataLoader(
        dataset=merged_data[:ii],
        batch_size=batchsize,
        shuffle=True,
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
     
def view_loader(data_loader):
    while 1:
      index = random.randint(0,len(data_loader.dataset)-1)
      data = data_loader.dataset[index]
      plt.subplot(1,2,1)
      image = np.zeros((34, 34, 3)) 
      histogram = np.transpose(event2histogram_mono(data[0]), (1, 2, 0))
      image[:,:,0:2] = histogram
      plt.imshow(image, cmap='magma')
      plt.title(data[-1])

      plt.subplot(1,2,2)
      plt.imshow(data[1].permute(1,2,0), cmap='magma')
      plt.title(data[-1])
      plt.show()

def train_latent():  

  N_MNIST_train_path = "./NMNIST_Train"
  MNIST_train_path = "./MNIST_Train"
  N_MNIST_test_path = "./NMNIST_Test"
  MNIST_test_path = "./MNIST_Test"
  batchsize = 32
  max_n_events = 2150
  seed = 42
  train_data_loader = create_loader(N_MNIST_train_path,MNIST_train_path,seed,batchsize,max_n_events,split=False)
  test_data_loader = create_loader(N_MNIST_test_path,MNIST_test_path,seed,batchsize,max_n_events,split=False)
  print(f"train_data_loader_size: {len(train_data_loader)*batchsize}")
  print(f"test_data_loader_size: {len(test_data_loader)*batchsize}")


  # init models
  device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
  print("cuda: "+ str(torch.cuda.is_available()))
  cnn_encoder = PointNetEncoder(max_n_events = max_n_events,input_dim=4).to(device)
  #cnn_encoder.load_state_dict(torch.load("cnn_encoder_18.pt",map_location='cuda:0'))
  cnn_encoder.load_state_dict(torch.load("cnn_encoder_34_2.pt",map_location='cuda:0'))
  image_encoder = imageEncoder(img_channels=1, num_layers=18, block=BasicBlock, num_classes=512).to(device)
   
  # init optimizer and loss
  params = list(image_encoder.parameters())
  optimizer = torch.optim.AdamW(params, lr=0.0001)
  loss_MSE = torch.nn.MSELoss()
  
  train_loss_list = []
  test_loss_list = []
  min_loss = 99999
  for epoch in range(30000):
    pbar_training = tqdm(total= (train_data_loader.batch_size*len(train_data_loader)) + (test_data_loader.batch_size*len(test_data_loader)))    
    avg_train_loss = 0
    for [N_MNIST, MNIST, input_maps, label] in train_data_loader:    
      N_MNIST = N_MNIST.to(device)
      N_MNIST_feat = cnn_encoder(N_MNIST)
      MNIST = MNIST.to(device)
      MNIST_feat = image_encoder(MNIST)     
      
      loss1 = loss_MSE(N_MNIST_feat,MNIST_feat)
      loss_tot_train = loss1 
      
      loss_tot_train.backward()
      optimizer.step()
      optimizer.zero_grad()
      pbar_training.update(train_data_loader.batch_size)
      avg_train_loss += loss1.item()/len(train_data_loader)
    avg_train_loss = np.round(avg_train_loss,12)
    with torch.no_grad():
      avg_test_loss = 0
      for [N_MNIST, MNIST, input_maps, label] in test_data_loader:    
        N_MNIST = N_MNIST.to(device)
        N_MNIST_feat = cnn_encoder(N_MNIST)
        MNIST = MNIST.to(device)
        MNIST_feat = image_encoder(MNIST)     
        
        loss1 = loss_MSE(N_MNIST_feat,MNIST_feat)
        loss_tot_train = loss1 
        
        pbar_training.update(test_data_loader.batch_size)
        avg_test_loss += loss1.item()/len(test_data_loader)
    avg_test_loss = np.round(avg_test_loss,12)
    train_loss_list.append(avg_train_loss)
    test_loss_list.append(avg_test_loss)
    if ((avg_test_loss)) < min_loss:
      min_loss = (avg_test_loss)
      torch.save(image_encoder.state_dict(), "image_encoder_tmp.pt")
    pbar_training.close()
    print("epoch:", epoch,"loss_train:",avg_train_loss,"loss_test:",avg_test_loss,"loss_sum:",min_loss) 

    if (epoch) % 25 == 0:
      torch.save(image_encoder.state_dict(), "image_encoder_end.pt")
      np.save("latent_loss_list_tmp.npy",[train_loss_list, test_loss_list])

def train_auto(index):  

  cwd = os.getcwd()
  print("Current Directory is: ", cwd)
  prefix = os.path.abspath(os.path.join(cwd, os.pardir))  

  '''
  N_MNIST_train_path = "./NMNIST_Train"
  MNIST_train_path = "./MNIST_Train"
  N_MNIST_test_path = "./NMNIST_Test"
  MNIST_test_path = "./MNIST_Test"
  batchsize = 32
  max_n_events = 2150
  seed = 42
  train_data_loader = create_loader(N_MNIST_train_path,MNIST_train_path,seed,batchsize,max_n_events,split=False)
  test_data_loader = create_loader(N_MNIST_test_path,MNIST_test_path,seed,batchsize,max_n_events,split=False)
  print(f"train_data_loader_size: {len(train_data_loader)*batchsize}")
  print(f"test_data_loader_size: {len(test_data_loader)*batchsize}")
  print(f"max_n_events: {max_n_events}") 
  
  
  '''
  N_MNIST_path = prefix + "/data/small_N_MNIST_training"
  MNIST_path = prefix + "/data/small_MNIST_img_training"
  batchsize = 32
  max_n_events = 1862
  seed = 42
  train_data_loader, test_data_loader= create_loader(N_MNIST_path,MNIST_path,seed,batchsize,max_n_events,split=True)
  print(f"train_data_loader_size: {len(train_data_loader)*batchsize}")
  print(f"test_data_loader_size: {len(test_data_loader)*batchsize}")

  

  # init models
  device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
  print("cuda: "+ str(torch.cuda.is_available()))
  event_encoder =  PointNetEncoder(max_n_events = max_n_events,input_dim=5).to(device).to(device)
  event_painter = PointNetDecoder(max_n_events = max_n_events,input_dim=5).to(device)

  # init optimizer and loss
  params = list(event_encoder.parameters()) + list(event_painter.parameters())
  optimizer = torch.optim.AdamW(params, lr=0.0001)
  loss_sinkhorn = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
  train_loss_list = []
  test_loss_list = []

  min_loss = 99999
  for epoch in range(30000):
    pbar_training = tqdm(total= (train_data_loader.batch_size*len(train_data_loader)) + (test_data_loader.batch_size*len(test_data_loader)))    
    avg_train_loss = 0
    for [N_MNIST, MNIST, input_maps, label] in train_data_loader:   
      N_MNIST = N_MNIST.to(device)
      label = label.to(device)
      vis_feat = event_encoder(N_MNIST)
      predict_event = event_painter(vis_feat)

      outputshape = torch.max(label[:,2])
      distance = (predict_event[:,:,4]-0.5)*(predict_event[:,:,4]-0.5)
      iDistance = torch.argsort(distance,dim=1,descending=True)
      iDistance = iDistance[:,:outputshape].to(device)

      predict_event_new = torch.zeros((batchsize,outputshape,5)).to(device)
      for i in range(batchsize):
        predict_event_new[i] = predict_event[i,iDistance[i],:]

      loss1 = loss_sinkhorn(predict_event_new,N_MNIST[:,:outputshape,:]).mean()

      optimizer.zero_grad()
      loss1.backward()
      optimizer.step()
      pbar_training.update(train_data_loader.batch_size)
      avg_train_loss += loss1.item()/len(train_data_loader)
    avg_train_loss = np.round(avg_train_loss,12)
    
    with torch.no_grad():
      avg_test_loss = 0
      for [N_MNIST, MNIST, input_maps, label] in test_data_loader:    
        N_MNIST = N_MNIST.to(device)
        label = label.to(device)
        vis_feat = event_encoder(N_MNIST)
        predict_event = event_painter(vis_feat)

        outputshape = torch.max(label[:,2])
        distance = ((predict_event[:,:,4]-0.5)*(predict_event[:,:,4]-0.5))
        iDistance = torch.argsort(distance,dim=1,descending=True)
        iDistance = iDistance[:,:outputshape].to(device)

        predict_event_new = torch.zeros((batchsize,outputshape,5)).to(device)
        for i in range(batchsize):
          predict_event_new[i] = predict_event[i,iDistance[i],:]

        loss1 = loss_sinkhorn(predict_event_new,N_MNIST[:,:outputshape,:]).mean()
        pbar_training.update(test_data_loader.batch_size)
        avg_test_loss += loss1.item()/len(test_data_loader)
    avg_test_loss = np.round(avg_test_loss,12)
    
    train_loss_list.append(avg_train_loss)
    test_loss_list.append(avg_test_loss)
    if ((avg_test_loss)) < min_loss:
      torch.save(event_encoder.state_dict(), prefix+"/models/cnn_encoder_"+str(index)+".pt")
      torch.save(event_painter.state_dict(), prefix+"/models/event_painter_"+str(index)+".pt")
      np.save(prefix+"/loss/loss_list_"+str(index)+".npy",[train_loss_list, test_loss_list])
      min_loss = (avg_test_loss)
    pbar_training.close()
    print("epoch:", epoch,"loss_train:",avg_train_loss,"loss_test:",avg_test_loss,"loss_sum:",min_loss) 

def train_getsize():  

  N_MNIST_train_path = "./NMNIST_Train"
  MNIST_train_path = "./MNIST_Train"
  N_MNIST_test_path = "./NMNIST_Test"
  MNIST_test_path = "./MNIST_Test"
  batchsize = 32
  max_n_events = 2150
  seed = 42
  train_data_loader = create_loader(N_MNIST_train_path,MNIST_train_path,seed,batchsize,max_n_events,split=False)
  test_data_loader = create_loader(N_MNIST_test_path,MNIST_test_path,seed,batchsize,max_n_events,split=False)
  print(f"train_data_loader_size: {len(train_data_loader)*batchsize}")
  print(f"test_data_loader_size: {len(test_data_loader)*batchsize}")

  # init models
  device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
  print("cuda: "+ str(torch.cuda.is_available()))
  #cnn_encoder = PointNetEncoder(max_n_events = max_n_events,input_dim=4).to(device)
  #cnn_encoder.load_state_dict(torch.load("cnn_encoder_18.pt",map_location='cuda:0'))
  #cnn_encoder.load_state_dict(torch.load("cnn_encoder_18.pt",map_location='cuda:0'))
  size_predicter = imageEncoder(img_channels=1, num_layers=18, block=BasicBlock, num_classes=1).to(device)
   
  # init optimizer and loss
  optimizer = torch.optim.AdamW(size_predicter.parameters(), lr=0.001)
  loss_MSE = torch.nn.MSELoss()
  
  train_loss_list = []
  test_loss_list = []
  min_loss = 99999
  for epoch in range(30000):
    pbar_training = tqdm(total= (train_data_loader.batch_size*len(train_data_loader)) + (test_data_loader.batch_size*len(test_data_loader)))    
    avg_train_loss = [0]
    for [N_MNIST, MNIST, input_maps, label] in train_data_loader:    
      MNIST = MNIST.to(device)
      label = label.to(device)
      
      ori_shape = (label[:,2]/max_n_events).unsqueeze(1)
      predicted_shape = size_predicter(MNIST)     
      loss1 = loss_MSE(ori_shape,predicted_shape)
      
      optimizer.zero_grad()
      loss1.backward()
      optimizer.step()
      pbar_training.update(train_data_loader.batch_size)
      avg_train_loss += np.round([loss1.item()/len(train_data_loader)],12)

    with torch.no_grad():
      avg_test_loss = [0]
      for [N_MNIST, MNIST, input_maps, label] in test_data_loader:    
        MNIST = MNIST.to(device)
        label = label.to(device)
        
        ori_shape = (label[:,2]/max_n_events).unsqueeze(1)
        predicted_shape = size_predicter(MNIST)     
        loss1 = loss_MSE(ori_shape,predicted_shape)
        
        pbar_training.update(test_data_loader.batch_size)
        avg_test_loss += np.round([loss1.item()/len(test_data_loader)],12)

    train_loss_list.append(avg_train_loss)
    test_loss_list.append(avg_test_loss)
    if (sum(avg_test_loss)) < min_loss:
      min_loss = sum(avg_test_loss)
      torch.save(size_predicter.state_dict(), "size_predicter_tmp.pt")
    pbar_training.close()
    print("epoch:", epoch,"loss_train:",avg_train_loss,"loss_test:",avg_test_loss,"loss_sum:",min_loss) 

    if (epoch) % 25 == 0:
      torch.save(size_predicter.state_dict(), "size_predicter_end_tmp.pt")
      np.save("size_predicter_loss_list_tmp.npy",[train_loss_list, test_loss_list])      
      
def delPad_th(predicted,target,label,thr):
  predicted = predicted[np.where(predicted[:,3] < thr)]
  sorted_indices = np.argsort(predicted[:,2])
  new_predicted = predicted[sorted_indices]  
  new_predicted[:,3] = torch.round(new_predicted[:,3]*1.5)
  
  new_target = target[:label[2]]
  new_target[:,3] = torch.round(new_target[:,3]*1.5)
  return new_predicted,new_target

def delPad_max(predicted,target,label,predicted_shape,marker):
  distance = (predicted[:,3]-marker)*(predicted[:,3]-marker)
  iDistance = np.argsort(-distance) #sort in descending order 
  iDistance = iDistance[:predicted_shape]
  predicted = predicted[iDistance]
  predicted[:,3] = torch.round(predicted[:,3])
    
  sorted_indices = np.argsort(predicted[:,2])
  predicted = predicted[sorted_indices]    
  #predicted = predicted/torch.amax(predicted)
  
  target = target[:label[2]]
  target[:,3] = torch.round(target[:,3])
  #target = target/torch.amax(target)
  return predicted,target

def delPad_similarity(predicted, target, label, predicted_shape):
  max_n_events = predicted.shape[0]
  distance_matrix = torch.cdist(predicted, predicted)
  distance_matrix = torch.triu(distance_matrix)
  distance_matrix[distance_matrix == 0] += torch.max(distance_matrix)
  max_indices = torch.topk(distance_matrix.flatten(), k=max_n_events*max_n_events).indices
  max_indices_2d = torch.unravel_index(max_indices, distance_matrix.shape)
  
  blackList = []
  whiteList = []
  ii = 0
  for ii in range(max_n_events*max_n_events):
    if len(blackList) == max_n_events-predicted_shape:
      break
    
    index_pair = [max_indices_2d[0][ii],max_indices_2d[1][ii]]
    state = [0,0]
    for i in range(2):
      if index_pair[i] in blackList:
        state[i] = 1
      if index_pair[i] in whiteList:
        state[i] = 2

    match state:
      case [0,0]:
        whiteList.append(index_pair[0])
        blackList.append(index_pair[1])
      case [0,1]:
        blackList.append(index_pair[0])  
      case [1,0]:
        blackList.append(index_pair[1])
      case [0,2]:
        blackList.append(index_pair[0])
      case [2,0]:
        blackList.append(index_pair[1])
      case [1,2]:
        1
      case [2,1]:
        1
      case [1,1]:
        1
      case [2,2]:
        whiteList.append(index_pair[0])
        blackList.append(index_pair[1])
      case _:
        print("ERROR")
  blackList = torch.tensor(blackList)
  mask = torch.ones_like(predicted[:,0], dtype=torch.bool)
  mask[blackList] = False
  new_predicted = predicted[mask,:]
  new_predicted[:,3] = torch.round(new_predicted[:,3])
  
  new_target = target[:label[2]]
  return new_predicted,new_target

def test_auto():   
  N_MNIST_test_path = "./NMNIST_Test"
  MNIST_test_path = "./MNIST_Test"
  batchsize = 16
  max_n_events = 2150
  seed = 42
  random.seed(seed)
  np.random.seed(seed)
  
  # ------------init models ------------
  device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
  print("cuda: "+ str(torch.cuda.is_available()))
  #cnn_encoder = PointNetEncoder(max_n_events = max_n_events,input_dim=4).to(device)

  event_encoder = PointNetEncoder(max_n_events = max_n_events,input_dim=4).to(device)
  event_painter = PointNetDecoder(max_n_events = max_n_events,input_dim=4).to(device)
  size_predicter = imageEncoder(img_channels=1, num_layers=18, block=BasicBlock, num_classes=1).to(device)
  
  #------------load data------------
  #cnn_encoder.load_state_dict(torch.load("cnn_encoder_16.pt",map_location='cuda:0'))
  event_painter.load_state_dict(torch.load("event_painter_34.pt",map_location='cuda:0'))
  event_encoder.load_state_dict(torch.load("cnn_encoder_34.pt",map_location='cuda:0'))
  size_predicter.load_state_dict(torch.load("size_predicter_26.pt",map_location='cuda:0'))
  
  loss_list = np.load("loss_list_34.npy")
  test_data_loader = create_loader(N_MNIST_test_path,MNIST_test_path,seed,batchsize,max_n_events,split=False)
  print(f"test_data_loader_size: {len(test_data_loader)*batchsize}")

  #------------run model------------
  predicted_nEvent_list = []
  target_nEvent_list = []
  with torch.no_grad():
    pbar_training = tqdm(total= ((test_data_loader.batch_size*len(test_data_loader))))    
    for [N_MNIST, MNIST, input_maps, label] in test_data_loader:    
      N_MNIST = N_MNIST.to(device)
      label = label.to(device)
      MNIST = MNIST.to(device)
      
      vis_feat = event_encoder(N_MNIST)
      predict_event = event_painter(vis_feat)

      #cal #event for target and predicted events
      predicted_shape = (size_predicter(MNIST)*max_n_events).to(torch.int)
      predicted_nEvent_list.extend(predicted_shape.tolist())
      target_nEvent = label[:,-1]
      target_nEvent_list.extend(target_nEvent.tolist())

      pbar_training.update(test_data_loader.batch_size)
    pbar_training.close()
  predict_event = predict_event.cpu()
  N_MNIST = N_MNIST.cpu()
  label = label.cpu()
  predicted_shape = predicted_shape.cpu()

  # ------------plot nEvent------------ 
  plt.figure()
  predicted_nEvent_list = np.array(predicted_nEvent_list)
  target_nEvent_list = np.array(target_nEvent_list)

  plt.subplot(1,2,1)
  plt.plot(predicted_nEvent_list,'b.',markersize=1,alpha=0.3,)    
  plt.plot(target_nEvent_list,'g.',markersize=1,alpha=0.3)        
  plt.ylabel("nEvent")
    
  plt.subplot(1,2,2)
  sorted_indices = np.argsort(target_nEvent_list)
  plt.plot(predicted_nEvent_list[sorted_indices],'b.',markersize=1,alpha=0.3)    
  plt.plot(target_nEvent_list[sorted_indices],'g.',markersize=1,alpha=0.3)    
  plt.ylabel("nEvent")
  plt.savefig('nEvent.png')  
  plt.figure().clear()
  print('nEvent.png DONE')
  
  # ------------plot parameters------------
  n = 6
  ii = 1
  plt.figure()
  for item in range(n):
    [predicted,target] = delPad_max(predict_event[item] ,N_MNIST[item],label[item],predicted_shape[item],0.5)

    plt.subplot(n,5,ii)
    plt.plot(target[:,0],'g.',markersize=1,alpha=0.3, label="Real_X")
    plt.plot((predicted[:,0]),'b.',markersize=1,alpha=0.3,label="Geneated_X")
    plt.title("x")
    #plt.legend()

    plt.subplot(n,5,ii+1)
    plt.plot(target[:,1], 'g.',markersize=1,alpha=0.3,label="Real_Y")
    plt.plot((predicted[:,1]),'b.',markersize=1,alpha=0.3,label="Geneated_Y")
    plt.title("y")
    #plt.legend()
    
    plt.subplot(n,5,ii+2)
    plt.plot((target[:,2]),'g.',markersize=1,alpha=0.3,label="Real_T")
    plt.plot((predicted[:,2]), 'b.',markersize=1,alpha=0.3,label="Geneated_T")
    #plt.title("Timestamp")

    predicted_p0 =  predicted[predicted[:,3] == 0] 
    target_p0 =  target[target[:,3] == 0] 
    plt.subplot(n,5,ii+3)
    plt.plot(target_p0[:,2],'g.',markersize=1,alpha=0.3,label="Real_T")
    plt.plot(predicted_p0[:,2], 'b.',markersize=1,alpha=0.3,label="Geneated_T")
    plt.title("p0")
    #plt.ylabel("Timestamp")

    predicted_p1 =  predicted[predicted[:,3] == 1] 
    target_p1 =  target[target[:,3] == 1] 
    plt.subplot(n,5,ii+4)
    plt.plot(target_p1[:,2],'g.',markersize=1,alpha=0.3,label="Real_T")
    plt.plot(predicted_p1[:,2], 'b.',markersize=1,alpha=0.3,label="Geneated_T")
    plt.title("p1")
    #plt.ylabel("Timestamp")

    #plt.legend()
    ii += 5
  #plt.tight_layout()
  plt.savefig('parameter.png')
  plt.figure().clear()
  print('parameter.png DONE')

  # ------------plot histogram------------
  plt.figure()
  ii = 1
  for item in range(n):
    [predicted,target] = delPad_max(predict_event[item],N_MNIST[item],label[item],predicted_shape[item],0.5)

    plt.subplot(n,2,ii)
    histogram_alt = event2histogram_tri(predicted)
    plt.imshow(histogram_alt,cmap='viridis')
    #plt.colorbar()
    plt.title("Generated")

    plt.subplot(n,2,ii+1)
    histogram = event2histogram_tri(target)
    plt.imshow(histogram,cmap='viridis')
    #plt.colorbar()
    plt.title("Real")
    ii += 2
  #plt.tight_layout()
  plt.savefig('histogram_new.png')
  plt.figure().clear()
  print('histogram_new.png DONE')
      

  # ------------plot loss------------
  plt.figure()
  plt.plot(loss_list[0,:],label = 'train')
  plt.plot(loss_list[1,:],label = 'test')
  plt.xlim([0,100])
  plt.legend()
  plt.tight_layout()
  plt.savefig('loss.png')
  print('loss.png DONE')
  
def test_latent():   
  N_MNIST_test_path = "./NMNIST_Test"
  MNIST_test_path = "./MNIST_Test"
  batchsize = 16
  max_n_events = 2150
  seed = 0

  # ------------init models ------------
  device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
  print("cuda: "+ str(torch.cuda.is_available()))
  #cnn_encoder = PointNetEncoder(max_n_events = max_n_events,input_dim=4).to(device)
  event_painter = PointNetDecoder(max_n_events = max_n_events,input_dim=4).to(device)
  image_encoder = imageEncoder(img_channels=1, num_layers=18, block=BasicBlock, num_classes=512).to(device)
  size_predicter = imageEncoder(img_channels=1, num_layers=18, block=BasicBlock, num_classes=1).to(device)

  #------------load data------------
  #cnn_encoder.load_state_dict(torch.load("cnn_encoder_16.pt",map_location='cuda:0'))
  event_painter.load_state_dict(torch.load("event_painter_34_2.pt",map_location='cuda:0'))
  image_encoder.load_state_dict(torch.load("image_encoder_34_2.pt",map_location='cuda:0'))
  size_predicter.load_state_dict(torch.load("size_predicter_26.pt",map_location='cuda:0'))
  loss_list = np.load("latent_loss_list_19.npy")
  test_data_loader = create_loader(N_MNIST_test_path,MNIST_test_path,seed,batchsize,max_n_events,split=False)
  print(f"test_data_loader_size: {len(test_data_loader)*batchsize}")

  #------------run model------------
  predicted_nEvent_list = []
  target_nEvent_list = []
  with torch.no_grad():
    pbar_training = tqdm(total= ((test_data_loader.batch_size*len(test_data_loader))))    
    for [N_MNIST, MNIST, input_maps, label] in test_data_loader:    
      N_MNIST = N_MNIST.to(device)
      label = label.to(device)
      MNIST = MNIST.to(device)

      vis_feat = image_encoder(MNIST)
      predict_event = event_painter(vis_feat)

      #cal #event for target and predicted events
      predicted_shape = (size_predicter(MNIST)*max_n_events).to(torch.int)
      predicted_nEvent_list.extend(predicted_shape.tolist())
      target_nEvent = label[:,-1]
      target_nEvent_list.extend(target_nEvent.tolist())

      pbar_training.update(test_data_loader.batch_size)
    pbar_training.close()
  predict_event = predict_event.cpu()
  N_MNIST = N_MNIST.cpu()
  MNIST = MNIST.cpu()
  label = label.cpu()
  predicted_shape = predicted_shape.cpu()

  # ------------plot nEvent------------ 
  plt.figure()
  predicted_nEvent_list = np.array(predicted_nEvent_list)
  target_nEvent_list = np.array(target_nEvent_list)

  plt.subplot(1,2,1)
  plt.plot(predicted_nEvent_list,'b.',markersize=1,alpha=0.3,)    
  plt.plot(target_nEvent_list,'g.',markersize=1,alpha=0.3)        
  plt.ylabel("nEvent")
    
  plt.subplot(1,2,2)
  sorted_indices = np.argsort(target_nEvent_list)
  plt.plot(predicted_nEvent_list[sorted_indices],'b.',markersize=1,alpha=0.3)    
  plt.plot(target_nEvent_list[sorted_indices],'g.',markersize=1,alpha=0.3)    
  plt.ylabel("nEvent")
  plt.savefig('nEvent.png')  
  plt.figure().clear()
  print('nEvent.png DONE')
  
  '''
  plt.figure()
  diff = target_nEvent_list[sorted_indices] - predicted_nEvent_list[sorted_indices]
  plt.hist(diff)
  plt.savefig('nEvent_diff.png')  
  plt.figure().clear()
  print('nEvent.png DONE')
  '''
  
  # ------------plot parameters------------
  n = 6
  ii = 1
  plt.figure()
  for item in range(n):
    [predicted,target] = delPad_max(predict_event[item] ,N_MNIST[item],label[item],predicted_shape[item],0.5)

    '''
    plt.subplot(n,5,ii)
    plt.plot(target[:,0],'g.',markersize=1,alpha=0.3, label="Real_X")
    plt.plot((predicted[:,0]),'b.',markersize=1,alpha=0.3,label="Geneated_X")
    plt.title("x")
    #plt.legend()

    plt.subplot(n,5,ii+1)
    plt.plot(target[:,1], 'g.',markersize=1,alpha=0.3,label="Real_Y")
    plt.plot((predicted[:,1]),'b.',markersize=1,alpha=0.3,label="Geneated_Y")
    plt.title("y")
    #plt.legend()
    
    '''
    plt.subplot(n,3,ii)
    plt.plot((target[:,2]),'g.',markersize=1,alpha=0.3,label="Real_T")
    plt.plot((predicted[:,2]), 'b.',markersize=1,alpha=0.3,label="Geneated_T")
    #plt.title("Timestamp")

    predicted_p0 =  predicted[predicted[:,3] == 0] 
    target_p0 =  target[target[:,3] == 0] 
    plt.subplot(n,3,ii+1)
    plt.plot(target_p0[:,2],'g.',markersize=1,alpha=0.3,label="Real_T")
    plt.plot(predicted_p0[:,2], 'b.',markersize=1,alpha=0.3,label="Geneated_T")
    plt.title("p0")
    #plt.ylabel("Timestamp")

    predicted_p1 =  predicted[predicted[:,3] == 1] 
    target_p1 =  target[target[:,3] == 1] 
    plt.subplot(n,3,ii+2)
    plt.plot(target_p1[:,2],'g.',markersize=1,alpha=0.3,label="Real_T")
    plt.plot(predicted_p1[:,2], 'b.',markersize=1,alpha=0.3,label="Geneated_T")
    plt.title("p1")
    #plt.ylabel("Timestamp")

    #plt.legend()
    ii += 3
  #plt.tight_layout()
  plt.savefig('parameter.png')
  plt.figure().clear()
  print('parameter.png DONE')

  # ------------plot histogram------------
  plt.figure()
  ii = 1
  for item in range(n):
    [predicted,target] = delPad_max(predict_event[item],N_MNIST[item],label[item],predicted_shape[item],0.5)

    plt.subplot(n,3,ii)
    histogram_alt = event2histogram_tri(predicted)
    plt.imshow(histogram_alt,cmap="hot")
    #plt.colorbar()
    plt.title("Generated")

    plt.subplot(n,3,ii+1)
    histogram = event2histogram_tri(target)
    plt.imshow(histogram,cmap="hot")
    #plt.colorbar()
    plt.title("Real")

    plt.subplot(n,3,ii+2)
    plt.imshow(MNIST[item][0],cmap="hot")
    #plt.colorbar()
    plt.title("Real")

    ii += 3
  #plt.tight_layout()
  plt.savefig('histogram_new.png')
  plt.figure().clear()
  print('histogram_new.png DONE')
      

  # ------------plot loss------------
  plt.figure()
  plt.plot(loss_list[0,:],label = 'train')
  plt.plot(loss_list[1,:],label = 'test')
  plt.xlim([0,100])
  plt.legend()
  plt.tight_layout()
  plt.savefig('loss.png')
  print('loss.png DONE')


if __name__ == "__main__":
  index = 50
  train_auto(index)
  #train_latent()
  #train_getsize()
  #test_auto()
  #test_latent()
  #test_new()
  



