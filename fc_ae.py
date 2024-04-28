import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type

class BasicBlock(nn.Module):
    """Basic Block for ResNet18 and ResNet34"""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, expansion: int = 1, downsample: nn.Module = None) -> None:
        super(BasicBlock, self).__init__()
        self.expansion = expansion
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels*self.expansion)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x # store copy of input tensor
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity # add input tensor to output tensor - residual connection
        out = self.relu(out)
        return out
    
class imageEncoder(nn.Module):
    def __init__(self, img_channels: int, num_layers: int, block: Type[BasicBlock], num_classes: int = 256) -> None:
        super(imageEncoder, self).__init__()
        if num_layers == 18:
            # The following `layers` list defines the number of `BasicBlock`
            # to use to build the network and how many basic blocks to stack
            # together.
            layers = [2, 2, 2, 2]
            self.expansion = 1
        self.in_channels = 64
        # All ResNets (18 to 152) contain a Conv2d => BN => ReLU for the first
        # three layers. Here, kernel size is 7.
        self.conv1 = nn.Conv2d(in_channels=img_channels, out_channels=self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block: Type[BasicBlock], out_channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels*self.expansion, kernel_size=1, stride=stride,bias=False),
                nn.BatchNorm2d(out_channels * self.expansion),
            )
        layers = [] # for storing the layers
        layers.append(block(self.in_channels, out_channels, stride, self.expansion, downsample))
        self.in_channels = out_channels * self.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, expansion=self.expansion))
        return nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #[T, N, C, H, W] [5,32,2,64,48]
        x = self.conv1(x) #[5,32,64,32,24]
        x = self.bn1(x) 
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # The spatial dimension of the final layer's feature
        # map should be (7, 7) for all ResNets.
        # print('Dimensions of the last convolutional feature map: ', x.shape)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class PointNetEncoder(nn.Module):
  def __init__(self, hidden_dim, max_n_events, input_dim):
      super().__init__()
      self.hidden_dim = hidden_dim
      self.max_n_events = max_n_events
      self.conv = nn.Sequential(
          nn.Conv1d(input_dim, 64, 1),
          nn.BatchNorm1d(64),
          nn.ReLU(),
          nn.Conv1d(64, 128, 1),
          nn.BatchNorm1d(128),
          nn.ReLU(),
          nn.Conv1d(128, 128, 1),
          nn.BatchNorm1d(128),
          nn.ReLU(),
          nn.Conv1d(128, 256, 1),
          nn.BatchNorm1d(256),
          nn.ReLU(),
          nn.Conv1d(256, self.hidden_dim, 1),
      )

  def forward(self, x):
      batch_size = x.shape[0]
      x = torch.permute(x, (0, 2, 1)) # (batch, 4, num_points)
      x = self.conv(x) #(batch, hidden_dim, num_points)
      x,_ = torch.max(x, dim=2) # (batch, hidden_dim)
      return x # output shape: (batch, dim)

class PointNetDecoder(nn.Module):
  def __init__(self, hidden_dim, max_n_events , input_dim):
      super().__init__()
      self.input_dim = input_dim
      self.hidden_dim = hidden_dim
      self.max_n_events = max_n_events
      self.fc = nn.Sequential(
          nn.Linear(self.hidden_dim, self.hidden_dim),
          nn.ReLU(),
          nn.Linear(self.hidden_dim, self.hidden_dim),
          nn.ReLU(),
          nn.Linear(self.hidden_dim, self.max_n_events*input_dim),
          nn.ReLU(),
          nn.Linear(self.max_n_events*input_dim, self.max_n_events*input_dim),
          nn.Sigmoid(),
      )

  def forward(self, x):
      """
      x: (batch, hidden_dim = 64)
      return: (batch, num_points = 390, 4)
      """
      batch_size = x.shape[0]
      x = self.fc(x) #(batch, num_points, hidden_dim)
      x = torch.reshape(x, (batch_size, self.max_n_events, self.input_dim)) # (batch, num_points, 4)
      return x # output shape: (batch, dim)