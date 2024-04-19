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
from dv import AedatFile

def load_Aedat(aedat_path):
    with AedatFile(aedat_path) as f:
        #f._name_info_map['events']['tsOffset']
        events = np.hstack([packet for packet in f['events'].numpy()])
        t, x, y, p = events['timestamp'], events['x'], events['y'], events['polarity']
        #print("duration[min]: " + str((t[-1]-t[0])/1E6/60))
        #print("load_Aedat completed")
        t = (t-t[0])/1E6
        x = x/128
        y = y/128

        index = np.abs(t-0.05).argmin()
        res = np.array([x[0:index],y[0:index],t[0:index],p[0:index]])
    return np.transpose(res)

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
  max_time_list = []
  for class_path in class_path_list:
      print(class_path)
      N_MNIST_class_path = os.path.join(N_MNIST_dir, class_path)
      #MNIST_class_path = os.path.join(MNIST_dir, class_path)
      file_path_list = os.listdir(N_MNIST_class_path)
      for file_path in file_path_list:        
        N_MNIST_file_path = os.path.join(N_MNIST_class_path, file_path)
        N_MNIST = load_Aedat(N_MNIST_file_path)
        ori_length = N_MNIST.shape[0]
        nEvent_list.append(ori_length)
        max_time_list.append(N_MNIST[-1,2])
        #if ori_length > max_n_events:
        #  print(N_MNIST_file_path)
        #  continue
        
        #N_MNIST = random_pad_with_c(N_MNIST,max_n_events)
        #N_MNIST = torch.tensor(N_MNIST).to(torch.float)
        N_MNIST_list.append(1)

        #MNIST_file_path = os.path.join(MNIST_class_path, modify_file_name(file_path))
        #MNIST = cv2.imread(MNIST_file_path)
        #MNIST = cv2.resize(MNIST, (34, 34), interpolation=cv2.INTER_LINEAR)/255 
        #MNIST = grayscale_transform(torch.tensor(MNIST).permute(2,0,1)).to(torch.float)
        MNIST_list.append(1)

        #event_histogram_data = torch.tensor(event2histogram_alt(N_MNIST))
        #input_map = torch.cat((MNIST, event_histogram_data), dim=0).to(torch.float32)
        #inputmap_list.append(input_map)

        inputmap_list.append(1)
        pad = int(ori_length < max_n_events)
        #label_list.append(torch.tensor([int(class_path),pad,ori_length]))
        label_list.append(1)

  np.save('nEvent_list.npy', nEvent_list)
  np.save('maxtime_list.npy', max_time_list)
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
      histogram = np.transpose(event2histogram_tri(data[0]), (1, 2, 0))
      image[:,:,0:2] = histogram
      plt.imshow(image, cmap='magma')
      plt.title(data[-1])

      plt.subplot(1,2,2)
      plt.imshow(data[1].permute(1,2,0), cmap='magma')
      plt.title(data[-1])
      plt.show()

def train_latent(autoencoder_index):  
  cwd = os.getcwd()
  print("Current Directory is: ", cwd)
  prefix = os.path.abspath(os.path.join(cwd, os.pardir))  

  #------------load data------------
  N_MNIST_train_path = prefix +"/data/NMNIST_Train"
  MNIST_train_path = prefix +"/data/MNIST_Train"
  N_MNIST_test_path = prefix +"/data/NMNIST_Test"
  MNIST_test_path = prefix +"/data/MNIST_Test"
  batchsize = 32
  max_n_events = 2150
  seed = 42
  train_data_loader = create_loader(N_MNIST_train_path,MNIST_train_path,seed,batchsize,max_n_events,split=False)
  test_data_loader = create_loader(N_MNIST_test_path,MNIST_test_path,seed,batchsize,max_n_events,split=False)
  print(f"train_data_loader_size: {len(train_data_loader)*batchsize}")
  print(f"test_data_loader_size: {len(test_data_loader)*batchsize}")
  print(f"max_n_events: {max_n_events}") 

  #------------init model------------
  device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
  print("cuda: "+ str(torch.cuda.is_available()))
  event_encoder = PointNetEncoder(max_n_events = max_n_events,input_dim=5).to(device)
  event_encoder.load_state_dict(torch.load(prefix+"/models/event_encoder_"+str(autoencoder_index)+".pt"))
  image_encoder = imageEncoder(img_channels=1, num_layers=18, block=BasicBlock, num_classes=512).to(device)
  params = list(image_encoder.parameters())
  optimizer = torch.optim.Adam(params, lr=0.0001)
  loss_MSE = torch.nn.MSELoss()
  
  #------------train model------------
  train_loss_list = []
  test_loss_list = []
  min_loss = 99999
  for epoch in range(30000):
    pbar_training = tqdm(total= (train_data_loader.batch_size*len(train_data_loader)) + (test_data_loader.batch_size*len(test_data_loader)))    
    avg_train_loss = 0
    for [N_MNIST, MNIST, input_maps, label] in train_data_loader:    
      N_MNIST = N_MNIST.to(device)
      N_MNIST_feat = event_encoder(N_MNIST)
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
        N_MNIST_feat = event_encoder(N_MNIST)
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
      torch.save(image_encoder.state_dict(), prefix+"/models/image_encoder_"+str(autoencoder_index)+".pt")
      min_loss = (avg_test_loss)
    pbar_training.close()
    print("epoch:", epoch,"loss_train:",avg_train_loss,"loss_test:",avg_test_loss,"loss_sum:",min_loss) 
    np.save(prefix+"/loss/latent_loss_list_"+str(autoencoder_index)+".npy",[train_loss_list, test_loss_list])

def train_auto(index):  
  cwd = os.getcwd()
  print("Current Directory is: ", cwd)
  prefix = os.path.abspath(os.path.join(cwd, os.pardir))  

  #------------load data------------
  N_CIFAR_path = prefix +"/data_new/CIFAR10DVS"
  batchsize = 32
  max_n_events = 2150
  seed = 42
  train_data_loader,test_data_loader = create_loader(N_CIFAR_path,"none",seed,batchsize,max_n_events,split=True)
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
  
  '''

  #------------init model------------
  device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
  print("cuda: "+ str(torch.cuda.is_available()))
  event_encoder = PointNetEncoder(max_n_events = max_n_events,input_dim=5).to(device)
  event_painter = PointNetDecoder(max_n_events = max_n_events,input_dim=5).to(device)
  params = list(event_encoder.parameters()) + list(event_painter.parameters())
  optimizer = torch.optim.Adam(params, lr=0.0001)
  loss_sinkhorn = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
  
  
  #------------train model------------
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

      '''
      outputshape = torch.max(label[:,2])
      distance = (predict_event[:,:,4]-0.5)*(predict_event[:,:,4]-0.5)
      iDistance = torch.argsort(distance,dim=1,descending=True)
      iDistance = iDistance[:,:outputshape].to(device)

      predict_event_new = torch.zeros((batchsize,outputshape,5)).to(device)
      for i in range(batchsize):
        predict_event_new[i] = predict_event[i,iDistance[i],:]

      loss1 = loss_sinkhorn(predict_event_new,N_MNIST[:,:outputshape,:]).mean()
      '''
      loss1 = loss_sinkhorn(predict_event,N_MNIST).mean()
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

        ''' 
        outputshape = torch.max(label[:,2])
        distance = ((predict_event[:,:,4]-0.5)*(predict_event[:,:,4]-0.5))
        iDistance = torch.argsort(distance,dim=1,descending=True)
        iDistance = iDistance[:,:outputshape].to(device)

        predict_event_new = torch.zeros((batchsize,outputshape,5)).to(device)
        for i in range(batchsize):
          predict_event_new[i] = predict_event[i,iDistance[i],:]

        loss1 = loss_sinkhorn(predict_event_new,N_MNIST[:,:outputshape,:]).mean()
        '''
        
     
        loss1 = loss_sinkhorn(predict_event,N_MNIST).mean()
        pbar_training.update(test_data_loader.batch_size)
        avg_test_loss += loss1.item()/len(test_data_loader)
    avg_test_loss = np.round(avg_test_loss,12)
    
    train_loss_list.append(avg_train_loss)
    test_loss_list.append(avg_test_loss)
    if ((avg_test_loss)) < min_loss:
      torch.save(event_encoder.state_dict(), prefix+"/models/event_encoder_"+str(index)+".pt")
      torch.save(event_painter.state_dict(), prefix+"/models/event_decoder_"+str(index)+".pt")
      
      min_loss = (avg_test_loss)
    pbar_training.close()
    print("epoch:", epoch,"loss_train:",avg_train_loss,"loss_test:",avg_test_loss,"loss_sum:",min_loss) 
    np.save(prefix+"/loss/loss_list_"+str(index)+".npy",[train_loss_list, test_loss_list])

def train_getsize(index):  
  cwd = os.getcwd()
  print("Current Directory is: ", cwd)
  prefix = os.path.abspath(os.path.join(cwd, os.pardir))  

  #------------load data------------
  N_MNIST_train_path = prefix +"/data/NMNIST_Train"
  MNIST_train_path = prefix +"/data/MNIST_Train"
  N_MNIST_test_path = prefix +"/data/NMNIST_Test"
  MNIST_test_path = prefix +"/data/MNIST_Test"
  batchsize = 32
  max_n_events = 2150
  seed = 42
  train_data_loader = create_loader(N_MNIST_train_path,MNIST_train_path,seed,batchsize,max_n_events,split=False)
  test_data_loader = create_loader(N_MNIST_test_path,MNIST_test_path,seed,batchsize,max_n_events,split=False)
  print(f"train_data_loader_size: {len(train_data_loader)*batchsize}")
  print(f"test_data_loader_size: {len(test_data_loader)*batchsize}")
  print(f"max_n_events: {max_n_events}") 

  # ------------init models ------------
  device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
  print("cuda: "+ str(torch.cuda.is_available()))
  size_predictor = imageEncoder(img_channels=1, num_layers=18, block=BasicBlock, num_classes=1).to(device)
  optimizer = torch.optim.Adam(size_predictor.parameters(), lr=0.001)
  loss_MSE = torch.nn.MSELoss()


  #------------run model------------
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
      predicted_shape = size_predictor(MNIST)     
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
        predicted_shape = size_predictor(MNIST)     
        loss1 = loss_MSE(ori_shape,predicted_shape)
        
        pbar_training.update(test_data_loader.batch_size)
        avg_test_loss += np.round([loss1.item()/len(test_data_loader)],12)

    train_loss_list.append(avg_train_loss)
    test_loss_list.append(avg_test_loss)
    if ((avg_test_loss)) < min_loss:
      torch.save(size_predictor.state_dict(), prefix+"/models/size_predictor_"+str(index)+".pt")
      min_loss = (avg_test_loss)
    pbar_training.close()
    print("epoch:", epoch,"loss_train:",avg_train_loss,"loss_test:",avg_test_loss,"loss_sum:",min_loss) 
    np.save(prefix+"/loss/size_predictor_loss_list_"+str(index)+".npy",[train_loss_list, test_loss_list])

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

def test_auto(autoencoder_index,size_predictor_index):   
  cwd = os.getcwd()
  print("Current Directory is: ", cwd)
  prefix = os.path.abspath(os.path.join(cwd, os.pardir))  

  #------------load data------------
  N_MNIST_test_path = prefix +"/data/NMNIST_Test"
  MNIST_test_path = prefix +"/data/MNIST_Test"
  batchsize = 32
  max_n_events = 2150
  seed = 42
  test_data_loader = create_loader(N_MNIST_test_path,MNIST_test_path,seed,batchsize,max_n_events,split=False)
  print(f"test_data_loader_size: {len(test_data_loader)*batchsize}")
  print(f"max_n_events: {max_n_events}") 
  
  # ------------init models ------------
  device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
  print("cuda: "+ str(torch.cuda.is_available()))
  event_encoder = PointNetEncoder(max_n_events = max_n_events,input_dim=5).to(device)
  event_decoder = PointNetDecoder(max_n_events = max_n_events,input_dim=5).to(device)
  size_predicter = imageEncoder(img_channels=1, num_layers=18, block=BasicBlock, num_classes=1).to(device)
  
  #------------load model------------
  event_encoder.load_state_dict(torch.load(prefix+"/models/event_encoder_"+str(autoencoder_index)+".pt",map_location='cuda:0'))
  event_decoder.load_state_dict(torch.load(prefix+"/models/event_decoder_"+str(autoencoder_index)+".pt",map_location='cuda:0'))
  size_predicter.load_state_dict(torch.load(prefix+"/models/size_predicter_"+str(size_predictor_index)+".pt",map_location='cuda:0'))  
  loss_list = np.load(prefix+"/loss/loss_list_"+str(autoencoder_index)+".npy")

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
      predict_event = event_decoder(vis_feat)

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
  plt.savefig(prefix + "/output/nEvent.png")  
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
  plt.savefig(prefix + '/output/parameter.png')
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
  plt.savefig(prefix + '/output/histogram_new.png')
  plt.figure().clear()
  print('histogram_new.png DONE')
      

  # ------------plot loss------------
  plt.figure()
  plt.plot(loss_list[0,:],label = 'train')
  plt.plot(loss_list[1,:],label = 'test')
  plt.xlim([0,100])
  plt.legend()
  plt.tight_layout()
  plt.savefig(prefix + '/output/loss.png')
  print('loss.png DONE')
  
def test_latent(autoencoder_index,size_predictor_index):   
  cwd = os.getcwd()
  print("Current Directory is: ", cwd)
  prefix = os.path.abspath(os.path.join(cwd, os.pardir))  
  
  #------------load data------------
  N_MNIST_test_path = prefix +"/data/NMNIST_Test"
  MNIST_test_path = prefix +"/data/MNIST_Test"
  batchsize = 32
  max_n_events = 2150
  seed = 42
  test_data_loader = create_loader(N_MNIST_test_path,MNIST_test_path,seed,batchsize,max_n_events,split=False)
  print(f"test_data_loader_size: {len(test_data_loader)*batchsize}")
  print(f"max_n_events: {max_n_events}") 

  # ------------init models ------------
  device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
  print("cuda: "+ str(torch.cuda.is_available()))
  event_decoder = PointNetDecoder(max_n_events = max_n_events,input_dim=5).to(device)
  image_encoder = imageEncoder(img_channels=1, num_layers=18, block=BasicBlock, num_classes=512).to(device)
  size_predicter = imageEncoder(img_channels=1, num_layers=18, block=BasicBlock, num_classes=1).to(device)

  #------------load model------------
  event_decoder.load_state_dict(torch.load(prefix+"/models/event_decoder_"+str(autoencoder_index)+".pt"))
  size_predicter.load_state_dict(torch.load(prefix+"/models/size_predicter_"+str(size_predictor_index)+".pt"))
  image_encoder.load_state_dict(torch.load(prefix+"/models/image_encoder_"+str(size_predictor_index)+".pt"))
  loss_list = np.load(prefix+"/loss/latent_loss_list_"+str(autoencoder_index)+".npy")

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
      predict_event = event_decoder(vis_feat)

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
  autoencoder_index = 100
  size_predictor_index = 50

  #train_getsize(size_predictor_index)
  train_auto(autoencoder_index)
  #train_latent(autoencoder_index)

  #test_auto(autoencoder_index,size_predictor_index)
  #test_latent(autoencoder_index,size_predictor_index)

  



