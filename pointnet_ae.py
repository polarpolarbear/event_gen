import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetEncoder(nn.Module):
  def __init__(self, hidden_dim = 512, max_n_events = 390):
      super().__init__()
      self.hidden_dim = hidden_dim
      self.max_n_events = max_n_events
      self.conv = nn.Sequential(
          nn.Conv1d(2, 64, 1),
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
  def __init__(self, hidden_dim = 512, max_n_events = 390):
      super().__init__()
      self.hidden_dim = hidden_dim
      self.max_n_events = max_n_events
      self.fc = nn.Sequential(
          nn.Linear(self.hidden_dim, 512),
          nn.ReLU(),
          nn.Linear(512, 512),
          nn.ReLU(),
          nn.Linear(512, self.max_n_events*2),
          nn.ReLU(),
          nn.Linear(self.max_n_events*2, self.max_n_events*2)
      )

  def forward(self, x):
      """
      x: (batch, hidden_dim = 64)
      return: (batch, num_points = 390, 4)
      """
      batch_size = x.shape[0]
      x = self.fc(x) #(batch, num_points, hidden_dim)
      x = torch.reshape(x, (batch_size, self.max_n_events, 2)) # (batch, num_points, 4)
      return x # output shape: (batch, dim)