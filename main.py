import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import cv2
from model import HistPredictor, CNNEncoder, EventPainter,Transformer,ModelPos,ModelP,ModelT
from fc_ae import PointNetEncoder, PointNetDecoder
from event_reading import read_event
import torchvision.transforms as transforms
import os
import random
from tqdm import tqdm
from geomloss import SamplesLoss  # See also ImagesLoss, VolumesLoss

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

def chamfer_distance_1D(y_pred, y):
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

def event2histogram(event_stream):
    hist = np.zeros((2, 34, 34))
    for event in event_stream.numpy():
        x = (33*event[0]).astype(int)
        y = (33*event[1]).astype(int)
        if event[2] == 1:
            hist[0, y, x] += 1
        else:
            hist[1, y, x] += 1
    return hist

def event2histogram_alt(event_stream):
    event_stream = event_stream.numpy()
    hist = np.zeros((2, 34, 34))
    xx = np.floor((event_stream[:,0]*33)).astype(int)
    yy= np.floor((event_stream[:,1]*33)).astype(int)
    pp = np.floor((event_stream[:,2])).astype(int)
    
    ii = np.where(pp == 2)
    if len(ii[0]) > 0 :
      np.add.at(hist, (pp[:ii[0][0]], yy[:ii[0][0]], xx[:ii[0][0]]), 1)
    else:
      np.add.at(hist, (pp, yy, xx), 1)
    return hist

def event2histogram_mono(event_stream):
    event_stream = event_stream.numpy()
    hist = np.zeros((34, 34))
    xx = np.floor((event_stream[:,0]*33)).astype(int)
    yy= np.floor((event_stream[:,1]*33)).astype(int)

    #xx[xx>1] = 1
    np.add.at(hist, (yy, xx), 1)
    return hist

def zeropad_N_MIST(N_MNIST,max_n_events):
  new_N_mist = np.zeros((max_n_events,4))
  ii = N_MNIST.shape[0] 
  new_N_mist[:ii ,:] = N_MNIST
  return new_N_mist
  
def last_pad_N_MIST(N_MNIST,max_n_events):
    n_event, _ = N_MNIST.shape
    padded_events = np.zeros((max_n_events, 4))  # 创建一个全零数组作为填充后的事件

    padded_events[:min(n_event, max_n_events)] = N_MNIST[:min(n_event, max_n_events)]  # 将原始事件填充到新的数组中

    if n_event < max_n_events:
        #indices = np.random.choice(n_event, max_n_events - n_event, replace=Flase)  # 从原始事件中随机选择事件的索引
        padded_events[n_event:] = N_MNIST[-1]  # 从原始事件中随机选择事件填充到新的数组中
        padded_events[n_event:,2] = 0

    return padded_events
  
def random_pad_N_MIST(N_MNIST,max_n_events):
    n_event, _ = N_MNIST.shape
    padded_events = np.zeros((max_n_events, 4))  # 创建一个全零数组作为填充后的事件

    padded_events[:min(n_event, max_n_events)] = N_MNIST[:min(n_event, max_n_events)]  # 将原始事件填充到新的数组中

    if n_event < max_n_events:
        indices = np.random.choice(n_event, max_n_events - n_event, replace=True)  # 从原始事件中随机选择事件的索引
        padded_events[n_event:] = N_MNIST[indices]  # 从原始事件中随机选择事件填充到新的数组中
        padded_events[n_event:,2] = 0

    return padded_events

def antipad_N_MIST(N_MNIST,max_n_events):
  ii = N_MNIST.shape[0] 
  nDelete = ii - max_n_events
  iDelete = np.linspace(0, ii-1, nDelete)
  iDelete = np.floor(iDelete).astype(int)
  myset = set(iDelete)
  if len(myset) == len(iDelete):
    N_MNIST1 = np.delete(N_MNIST,iDelete,axis=0)
    return N_MNIST1
  else:
    print("error")
    return N_MNIST

def create_loader(N_MNIST_dir, MNIST_dir,seed,batchsize,max_n_events,split):
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
      #if int(class_path) < 2:
      #  break

      N_MNIST_class_path = os.path.join(N_MNIST_dir, class_path)
      MNIST_class_path = os.path.join(MNIST_dir, class_path)
      file_path_list = os.listdir(N_MNIST_class_path)
      for file_path in file_path_list:        
        N_MNIST_file_path = os.path.join(N_MNIST_class_path, file_path)
        N_MNIST = read_event(N_MNIST_file_path)

        nEvent_list.append(N_MNIST.shape[0])
        pad = int(N_MNIST.shape[0] < max_n_events)
                  
        N_MNIST = random_pad_N_MIST(N_MNIST,max_n_events)
        N_MNIST = torch.tensor(N_MNIST).to(torch.float32)
        N_MNIST_list.append(N_MNIST)

        #MNIST_file_path = os.path.join(MNIST_class_path, modify_file_name(file_path))
        #MNIST = cv2.imread(MNIST_file_path)
        #MNIST = cv2.resize(MNIST, (34, 34), interpolation=cv2.INTER_LINEAR)/255 
        #MNIST = grayscale_transform(torch.tensor(MNIST).permute(2,0,1)).to(torch.float32)
        MNIST_list.append(1)

        #event_histogram_data = torch.tensor(event2histogram_alt(N_MNIST))
        #input_map = torch.cat((MNIST, event_histogram_data), dim=0).to(torch.float32)
        #inputmap_list.append(input_map)

        inputmap_list.append(1)
        label_list.append([int(class_path),pad])

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
      histogram = np.transpose(event2histogram(data[0]), (1, 2, 0))
      image[:,:,0:2] = histogram
      plt.imshow(image, cmap='magma')
      plt.title(data[-1])

      plt.subplot(1,2,2)
      plt.imshow(data[1].permute(1,2,0), cmap='magma')
      plt.title(data[-1])
      plt.show()

def train():  
  #SMALL
  '''
  N_MNIST_path = "./N_MNIST_small_training"
  MNIST_path = "./MNIST_img_small_training"
  batchsize = 16
  min_n_events = 165
  max_n_events = 160
  seed = 42
  train_data_loader, test_data_loader= create_loader(N_MNIST_path,MNIST_path,seed,batchsize,min_n_events,max_n_events,split=True)
  print(f"train_data_loader_size: {len(train_data_loader)*batchsize}")
  print(f"test_data_loader_size: {len(test_data_loader)*batchsize}")
  
  '''
  N_MNIST_train_path = "./medium_NMNIST_Train"
  MNIST_train_path = "./medium_MNIST_Train"
  N_MNIST_test_path = "./medium_NMNIST_Test"
  MNIST_test_path = "./medium_MNIST_Test"
  batchsize = 32
  min_n_events = 860
  max_n_events = 2100
  seed = 42
  train_data_loader = create_loader(N_MNIST_train_path,MNIST_train_path,seed,batchsize,min_n_events,max_n_events,split=False)
  test_data_loader = create_loader(N_MNIST_test_path,MNIST_test_path,seed,batchsize,min_n_events,max_n_events,split=False)
  print(f"train_data_loader_size: {len(train_data_loader)*batchsize}")
  print(f"test_data_loader_size: {len(test_data_loader)*batchsize}")
 


  #view_loader(train_data_loader)

  # init models
  device = torch.device("cuda:4") if torch.cuda.is_available() else torch.device("cpu")
  print("cuda: "+ str(torch.cuda.is_available()))

  cnn_encoder = CNNEncoder().to(device)
  event_painter = EventPainter(max_n_events = max_n_events).to(device)
  #cnn_encoder.load_state_dict(torch.load("cnn_encoder.pt"))
  #event_painter.load_state_dict(torch.load("event_painter.pt"))

  # init optimizer and loss
  params = list(cnn_encoder.parameters()) + list(event_painter.parameters())
  optimizer = torch.optim.SGD(params, lr=0.0001)
  
  #loss_polarity = nn.CrossEntropyLoss()
  #loss_time = nn.MSELoss()
  loss_spatial = nn.MSELoss()
  
  loss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)


  train_loss_list = []
  test_loss_list = []

  min_loss = 99999
  for epoch in range(30):
    pbar_training = tqdm(total= (train_data_loader.batch_size*len(train_data_loader)) + (test_data_loader.batch_size*len(test_data_loader)))    
    avg_train_loss = [0,0]
    for [N_MNIST, MNIST, input_maps, label] in train_data_loader:    
      x_maps = MNIST.to(device)
      N_MNIST = N_MNIST.to(device)

      vis_feat = cnn_encoder(x_maps)
      predict_event = event_painter(vis_feat)

      #loss_xy_train = chamfer_distance(predict_event, y_event)
      #label_onehot = F.one_hot(y_event[:,:,2].to(torch.int64), 2).float()
      #loss_p = loss_polarity(predict_event[:,:,], label_onehot)

      loss_xy = loss_spatial(predict_event[:,:,:2], N_MNIST[:,:,:2])
      #loss_t_train = loss_time(y_event[:,:,:2],predict_event[:,:,:2])*10
  
      loss_sinkhorn =loss(predict_event[:,:,:2],N_MNIST[:,:,:2]).mean()

      loss_cd = chamfer_distance(predict_event[:,:,:2],N_MNIST[:,:,:2])

      loss_tot_train = loss_sinkhorn + loss_xy + loss_cd
      loss_tot_train.backward()

      optimizer.step()
      optimizer.zero_grad()
      pbar_training.update(train_data_loader.batch_size)
      avg_train_loss += np.round([loss_sinkhorn.item()/len(train_data_loader),loss_cd.item()/len(train_data_loader) ],6)

    with torch.no_grad():
      avg_test_loss = [0,0]
      for [N_MNIST, MNIST, input_maps, label] in test_data_loader:    
        x_maps = MNIST.to(device)
        N_MNIST = N_MNIST.to(device)
        vis_feat = cnn_encoder(x_maps)
        predict_event = event_painter(vis_feat)
        
        #loss_xy_train = chamfer_distance(predict_event, y_event) 
        #loss_xy = loss_spatial(predict_event[:,:,:2], y_event[:,:,:2])*10
        #loss_t_train = loss_time(y_event[:,:,:2],predict_event[:,:,:2])*10
        loss_xy = loss_spatial(predict_event[:,:,:2], N_MNIST[:,:,:2])
        loss_sinkhorn =loss(predict_event[:,:,:2],N_MNIST[:,:,:2]).mean()
        
        pbar_training.update(test_data_loader.batch_size)
        avg_test_loss += np.round([loss_sinkhorn.item()/len(test_data_loader), loss_xy.item()/len(test_data_loader)],4)

    train_loss_list.append(avg_train_loss)
    test_loss_list.append(avg_test_loss)
    if (sum(avg_test_loss)) < min_loss:
      torch.save(cnn_encoder.state_dict(), "cnn_encoder.pt")
      torch.save(event_painter.state_dict(), "event_painter.pt")
      min_loss = sum(avg_test_loss)
    pbar_training.close()
    print("epoch:", epoch,"loss_train:",avg_train_loss,"loss_test:",avg_test_loss,"loss_sum:",min_loss) #,"loss histogram:",loss_histogram.item())\


    if (epoch+1) % 100 == 0:
      torch.save(cnn_encoder.state_dict(), "cnn_encoder.pt")
      torch.save(event_painter.state_dict(), "event_painter.pt")
      np.save("loss_train.npy", train_loss_list)
      np.save("loss_test.npy", test_loss_list)
      np.save("loss_list.npy",[train_loss_list, test_loss_list])

def train_auto():  
  #SMALL

  '''
  N_MNIST_path = "./small_N_MNIST_training"
  MNIST_path = "./small_MNIST_img_training"
  batchsize = 16
  min_n_events = 0
  max_n_events = 1862
  seed = 42
  train_data_loader, test_data_loader= create_loader(N_MNIST_path,MNIST_path,seed,batchsize,min_n_events,max_n_events,split=True)
  print(f"train_data_loader_size: {len(train_data_loader)*batchsize}")
  print(f"test_data_loader_size: {len(test_data_loader)*batchsize}")
  
  '''
  N_MNIST_train_path = "./NMNIST_Train"
  MNIST_train_path = "./MNIST_Train"
  N_MNIST_test_path = "./NMNIST_Test"
  MNIST_test_path = "./MNIST_Test"
  batchsize = 32
  max_n_events = 3000
  seed = 42
  train_data_loader = create_loader(N_MNIST_train_path,MNIST_train_path,seed,batchsize,max_n_events,split=False)
  test_data_loader = create_loader(N_MNIST_test_path,MNIST_test_path,seed,batchsize,max_n_events,split=False)
  print(f"train_data_loader_size: {len(train_data_loader)*batchsize}")
  print(f"test_data_loader_size: {len(test_data_loader)*batchsize}")


  #view_loader(train_data_loader)

  # init models
  device = torch.device("cuda:4") if torch.cuda.is_available() else torch.device("cpu")
  print("cuda: "+ str(torch.cuda.is_available()))

  cnn_encoder = PointNetEncoder(max_n_events = max_n_events).to(device)
  event_painter = PointNetDecoder(max_n_events = max_n_events).to(device)
  #cnn_encoder.load_state_dict(torch.load("cnn_encoder.pt"))
  #event_painter.load_state_dict(torch.load("event_painter.pt"))

  # init optimizer and loss
  params = list(cnn_encoder.parameters()) + list(event_painter.parameters())
  optimizer = torch.optim.Adam(params, lr=0.0001)
  
  #loss_polarity = nn.CrossEntropyLoss()
  #loss_time = nn.MSELoss()
  #loss_spatial = nn.MSELoss()
  
  loss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)


  train_loss_list = []
  test_loss_list = []

  min_loss = 99999
  for epoch in range(30000):
    pbar_training = tqdm(total= (train_data_loader.batch_size*len(train_data_loader)) + (test_data_loader.batch_size*len(test_data_loader)))    
    avg_train_loss = [0]
    for [N_MNIST, MNIST, input_maps, label] in train_data_loader:    
      #x_maps = MNIST.to(device)
      N_MNIST = N_MNIST.to(device)
      vis_feat = cnn_encoder(N_MNIST[:,:,:3])
      predict_event = event_painter(vis_feat)

      #loss_xy_train = chamfer_distance(predict_event, y_event)
      #label_onehot = F.one_hot(y_event[:,:,2].to(torch.int64), 2).float()
      #loss_p = loss_polarity(predict_event[:,:,], label_onehot)

      #loss_xy = loss_spatial(predict_event[:,:,:2], N_MNIST[:,:,:2])
      #loss_t_train = loss_time(y_event[:,:,:2],predict_event[:,:,:2])*10
  
      loss_sinkhorn = loss(predict_event[:,:,:3],N_MNIST[:,:,:3]).mean()

      #loss_cd = chamfer_distance(predict_event[:,:,2],N_MNIST[:,:,2])

      #histogram = event2histogram_mono_torcch(predict_event)

      loss_tot_train = loss_sinkhorn 
      
      loss_tot_train.backward()

      optimizer.step()
      optimizer.zero_grad()
      pbar_training.update(train_data_loader.batch_size)
      avg_train_loss += np.round([loss_sinkhorn.item()/len(train_data_loader)],8)

    with torch.no_grad():
      avg_test_loss = [0]
      for [N_MNIST, MNIST, input_maps, label] in test_data_loader:    
        #x_maps = MNIST.to(device)
        N_MNIST = N_MNIST.to(device)
        vis_feat = cnn_encoder(N_MNIST[:,:,:3])
        predict_event = event_painter(vis_feat)
        
        #loss_xy_train = chamfer_distance(predict_event, y_event) 
        #loss_xy = loss_spatial(predict_event[:,:,:2], y_event[:,:,:2])*10
        #loss_t_train = loss_time(y_event[:,:,:2],predict_event[:,:,:2])*10
        #loss_xy = loss_spatial(predict_event[:,:,:3], N_MNIST[:,:,:3])
        loss_sinkhorn =loss(predict_event[:,:,:3],N_MNIST[:,:,:3]).mean()
        
        
        pbar_training.update(test_data_loader.batch_size)
        avg_test_loss += np.round([loss_sinkhorn.item()/len(test_data_loader)],8)


    train_loss_list.append(avg_train_loss)
    test_loss_list.append(avg_test_loss)
    if (sum(avg_test_loss)) < min_loss:
      torch.save(cnn_encoder.state_dict(), "cnn_encoder.pt")
      torch.save(event_painter.state_dict(), "event_painter.pt")
      min_loss = sum(avg_test_loss)
    pbar_training.close()
    print("epoch:", epoch,"loss_train:",avg_train_loss,"loss_test:",avg_test_loss,"loss_sum:",min_loss) #,"loss histogram:",loss_histogram.item())\

    if (epoch) % 50 == 0:
      torch.save(cnn_encoder.state_dict(), "cnn_encoder_end.pt")
      torch.save(event_painter.state_dict(), "event_painter_end.pt")
      np.save("loss_list.npy",[train_loss_list, test_loss_list])

def test():   
  N_MNIST_train_path = "./medium_NMNIST_Train"
  MNIST_train_path = "./medium_MNIST_Train"
  N_MNIST_test_path = "./medium_NMNIST_Test"
  MNIST_test_path = "./medium_MNIST_Test"
  batchsize = 32
  min_n_events = 1255
  max_n_events = 1250
  seed = 42
  train_data_loader = create_loader(N_MNIST_train_path,MNIST_train_path,seed,batchsize,min_n_events,max_n_events,split=False)
  test_data_loader = create_loader(N_MNIST_test_path,MNIST_test_path,seed,batchsize,min_n_events,max_n_events,split=False)
  print(f"train_data_loader_size: {len(train_data_loader)*batchsize}")
  print(f"test_data_loader_size: {len(test_data_loader)*batchsize}")
  # init models
  cnn_encoder = CNNEncoder()
  event_painter = EventPainter(max_n_events = max_n_events)
  cnn_encoder.load_state_dict(torch.load("cnn_encoder_10.pt"))
  event_painter.load_state_dict(torch.load("event_painter_10.pt"))
  
  with torch.no_grad():
    index = random.randint(0,len(train_data_loader.dataset)-1)
    index = 5
    print(f"index:{index}")
    [y_event, MNIST, input_maps, label] = train_data_loader.dataset[index]
    x_maps = MNIST.reshape(1,MNIST.shape[0],MNIST.shape[1],MNIST.shape[2])
    vis_feat = cnn_encoder(x_maps)
    predict_event = event_painter(vis_feat)
    #predict_event[0,:,2] = torch.round(predict_event[0,:,2])
    np.save("y_event.npy",y_event.numpy())
    np.save("predict_event.npy",predict_event.numpy())

  plt.subplot(2,2,1)
  plt.plot(y_event[:,0]*33, label="Real_X")
  plt.plot(predict_event[0,:,0]*33, alpha=0.6,label="Geneated_X")
  plt.title("x")
  plt.legend()

  plt.subplot(2,2,2)
  plt.plot(y_event[:,1]*33, label="Real_Y")
  plt.plot(predict_event[0,:,1]*33, alpha=0.6,label="Geneated_Y")
  plt.title("y")
  plt.legend()

  '''
  #Polarity
  plt.subplot(2,2,3)
  plt.plot(y_event[:,2], '.',label="Real_P")
  ii = np.where(predict_event[0,:,2] == 2)
  acc = sum(predict_event[0,:,2] == y_event[:,2])/len(y_event[:,2])
  acc = np.round(acc.item(),4)
  if len(ii[0]) > 0 :
    acc1 = sum(predict_event[0,:ii[0][0],2] == y_event[:ii[0][0],2])/len(y_event[:,2])
    acc1 = np.round(acc1.item(),4)
  else:
    acc1 = acc

  #print((predict_event[0,:,0]*33))

  plt.plot(predict_event[0,:,2], '.',alpha=0.6,label="Geneated_P")
  print(acc1)
  print(acc)
  plt.title(f"p: acc = {acc}, acc1 = {acc1}")
  plt.legend()
  '''
  
  plt.subplot(2,2,4)
  #plt.plot(y_event[:,2],'.',label="Real_T")
  plt.plot(predict_event[0,:,2], '.',alpha=0.6,label="Geneated_T")
  plt.title("Timestamp")
  plt.legend()

  plt.tight_layout()
  plt.savefig('parameter.png')
  plt.figure().clear()

  plt.subplot(1,3,1)
  histogram_alt = event2histogram_mono(predict_event[0])
  #image = np.zeros((34, 34, 3)) 
  #histogram = np.transpose(histogram_alt, (1, 2, 0))
  #image[:,:,0:2] = histogram
  plt.imshow(histogram_alt, cmap='magma')
  plt.colorbar()
  plt.title("Generated_alt")

  plt.subplot(1,3,2)
  histogram = event2histogram_mono(y_event)
  #image = np.zeros((34, 34, 3)) 
  #histogram = np.transpose(histogram, (1, 2, 0))
  #image[:,:,0:2] = histogram
  plt.imshow(histogram, cmap='magma')
  plt.colorbar()
  plt.title("Real")

  plt.subplot(1,3,3)
  plt.imshow(MNIST[0], cmap='magma')
  plt.title("MNIST")

  plt.tight_layout()
  plt.savefig('histogram.png')
  plt.figure().clear()

  train_loss_list = np.load("loss_list.npy")
  plt.plot(train_loss_list[0,:],label = 'train')
  plt.plot(train_loss_list[1,:],label = 'test')
  #plt.plot(train_loss_list[0,:,1],label = 'p')
  #plt.plot(train_loss_list[0,:,2],label = 't')

  plt.xlim([0,100])
  plt.legend()
  plt.tight_layout()
  plt.savefig('loss.png')

def test_auto():   
 #SMALL
  '''
  N_MNIST_path = "./small_N_MNIST_training"
  MNIST_path = "./small_MNIST_img_training"
  batchsize = 16
  min_n_events = 100
  max_n_events = 390
  seed = 0
  train_data_loader, test_data_loader= create_loader(N_MNIST_path,MNIST_path,seed,batchsize,min_n_events,max_n_events,split=True)
  print(f"train_data_loader_size: {len(train_data_loader)*batchsize}")
  print(f"test_data_loader_size: {len(test_data_loader)*batchsize}")
  
  '''
  N_MNIST_train_path = "./medium_NMNIST_Train"
  MNIST_train_path = "./medium_MNIST_Train"
  N_MNIST_test_path = "./medium_NMNIST_Test"
  MNIST_test_path = "./medium_MNIST_Test"
  batchsize = 16
  min_n_events = 0
  max_n_events = 1862
  seed = 0
  train_data_loader = create_loader(N_MNIST_train_path,MNIST_train_path,seed,batchsize,min_n_events,max_n_events,split=False)
  test_data_loader = create_loader(N_MNIST_test_path,MNIST_test_path,seed,batchsize,min_n_events,max_n_events,split=False)
  print(f"train_data_loader_size: {len(train_data_loader)*batchsize}")
  print(f"test_data_loader_size: {len(test_data_loader)*batchsize}")

  # init models
  cnn_encoder = PointNetEncoder(max_n_events = max_n_events)
  event_painter = PointNetDecoder(max_n_events = max_n_events)
  cnn_encoder.load_state_dict(torch.load("cnn_encoder_10.pt"))
  event_painter.load_state_dict(torch.load("event_painter_10.pt"))
  train_loss_list = np.load("loss_list_10.npy")
  
  with torch.no_grad():
    index = 0
    print(f"index:{index}")
    [N_MNIST, MNIST, input_maps, label] = test_data_loader.dataset[index]
    N_MNIST = N_MNIST.reshape(1,N_MNIST.shape[0],N_MNIST.shape[1])
    
    vis_feat = cnn_encoder(N_MNIST[:,:,:3])
    predict_event = event_painter(vis_feat)
    #predict_event[0,:,2] = torch.round(predict_event[0,:,2])
    #np.save("y_event.npy",N_MNIST.numpy())
    #np.save("predict_event.npy",predict_event.numpy())


  sorted_indices = np.argsort(predict_event[0,:,2])

  # Sort B and A using the sorted indices
  plt.subplot(2,2,1)
  plt.plot(N_MNIST[0,:,0]*33, label="Real_X")
  plt.plot((predict_event[0,:,0]*33)[sorted_indices], alpha=0.6,label="Geneated_X")
  plt.title("x")
  plt.legend()

  plt.subplot(2,2,2)
  plt.plot(N_MNIST[0,:,1]*33, label="Real_Y")
  plt.plot((predict_event[0,:,1]*33)[sorted_indices], alpha=0.6,label="Geneated_Y")
  plt.title("y")
  plt.legend()

  '''
  #Polarity
  plt.subplot(2,2,3)
  plt.plot(y_event[:,2], '.',label="Real_P")
  ii = np.where(predict_event[0,:,2] == 2)
  acc = sum(predict_event[0,:,2] == y_event[:,2])/len(y_event[:,2])
  acc = np.round(acc.item(),4)
  if len(ii[0]) > 0 :
    acc1 = sum(predict_event[0,:ii[0][0],2] == y_event[:ii[0][0],2])/len(y_event[:,2])
    acc1 = np.round(acc1.item(),4)
  else:
    acc1 = acc

  #print((predict_event[0,:,0]*33))

  plt.plot(predict_event[0,:,2], '.',alpha=0.6,label="Geneated_P")
  print(acc1)
  print(acc)
  plt.title(f"p: acc = {acc}, acc1 = {acc1}")
  plt.legend()
  '''
  
  plt.subplot(2,2,4)
  plt.plot(np.sort(N_MNIST[0,:,2]),'.',label="Real_T")
  plt.plot((predict_event[0,:,2])[sorted_indices], '.',alpha=0.6,label="Geneated_T")
  plt.title("Timestamp")
  plt.legend()

  plt.tight_layout()
  plt.savefig('parameter.png')
  plt.figure().clear()

  plt.subplot(1,3,1)
  histogram_alt = event2histogram_mono(predict_event[0])
  #image = np.zeros((34, 34, 3)) 
  #histogram = np.transpose(histogram_alt, (1, 2, 0))
  #image[:,:,0:2] = histogram
  plt.imshow(histogram_alt, cmap='magma')
  plt.colorbar()
  plt.title("Generated_alt")

  plt.subplot(1,3,2)
  histogram = event2histogram_mono(N_MNIST[0])
  #image = np.zeros((34, 34, 3)) 
  #histogram = np.transpose(histogram, (1, 2, 0))
  #image[:,:,0:2] = histogram
  plt.imshow(histogram, cmap='magma')
  plt.colorbar()
  plt.title("Real")

  plt.subplot(1,3,3)
  plt.imshow(MNIST[0], cmap='magma')
  plt.title("MNIST")

  plt.tight_layout()
  plt.savefig('histogram.png')
  plt.figure().clear()


  plt.plot(train_loss_list[0,:],label = 'train')
  plt.plot(train_loss_list[1,:],label = 'test')
  #plt.plot(train_loss_list[0,:,1],label = 'p')
  #plt.plot(train_loss_list[0,:,2],label = 't')

  plt.xlim([0,100])
  plt.legend()
  plt.tight_layout()
  plt.savefig('loss.png')



if __name__ == "__main__":
  train_auto()
  #test_auto()
  



