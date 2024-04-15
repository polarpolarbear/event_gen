import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class HistPredictor(nn.Module):
  def __init__(self,n_maps = 32):
    super().__init__()
    """
    input: Image (grey scale)
    output: predicted event histogram (+ and - 2 channels)
    """
    self.conv_blocks_in = nn.Sequential(
        nn.Conv2d(1, n_maps, 3, 1, 1),
        nn.BatchNorm2d(n_maps),
        nn.ReLU(),
        nn.Conv2d(n_maps, n_maps, 3, 1, 1),
        nn.BatchNorm2d(n_maps),
        nn.ReLU()
    )

    self.conv_block_2 = nn.Sequential(
        nn.Conv2d(n_maps, n_maps, 3, 1, 1),
        nn.BatchNorm2d(n_maps),
        nn.ReLU(),
        nn.Conv2d(n_maps, n_maps, 3, 1, 1),
        nn.BatchNorm2d(n_maps),
    )

    self.conv_block_3 = nn.Sequential(
        nn.Conv2d(n_maps, n_maps, 3, 1, 1),
        nn.BatchNorm2d(n_maps),
        nn.ReLU(),
        nn.Conv2d(n_maps, n_maps, 3, 1, 1),
        nn.BatchNorm2d(n_maps),
    )

    self.conv_block_out = nn.Sequential(
        nn.Conv2d(n_maps, n_maps, 3, 1, 1),
        nn.BatchNorm2d(n_maps),
        nn.ReLU(),
        nn.Conv2d(n_maps, 2, 3, 1, 1),
        nn.BatchNorm2d(2),
        nn.ReLU()
    )

  def forward(self, x):
      x = self.conv_blocks_in(x)
      x = F.relu(x + self.conv_block2(x))
      x = F.relu(x + self.conv_block3(x))
      x = self.conv_block_out(x)
      return x

class CNNEncoder(nn.Module):
  def __init__(self,hidden_dim = 64):
    super().__init__()
    self.hidden_dim = hidden_dim
    """
    input: current event histogram (2) and target event histgram (2)
    output: embedding of every pixels
    """
    self.conv_blocks_in = nn.Sequential(
        nn.Conv2d(1, 16, 3, 1, 1), #input 3 for single image, 4 for video frames
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, 1, 1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, hidden_dim, 3, 1, 1),
        nn.BatchNorm2d(hidden_dim),
        nn.ReLU(),
    )

  def forward(self,x):
    batch_size = x.shape[0]
    x = self.conv_blocks_in(x)
    x = torch.permute(x, (0, 2, 3, 1))
    h, w = x.shape[1], x.shape[2]
    num_feat_map = x.shape[3]
    x = x.reshape(batch_size, h*w, num_feat_map)
    return x

class EventPainter(nn.Module):
  def __init__(self, hidden_dim = 64, n_heads=8, n_enc_layers = 3, n_dec_layers=3, max_n_events = 512, feat_map_h = 34, feat_map_w = 34):
      super().__init__()
      """
      check:
      feature map width and height
      """
      self.query_pos = nn.Parameter(torch.rand(max_n_events, hidden_dim)) #to consider: use learnable or fixed?
      self.pos_encoding  = nn.Parameter(torch.rand(feat_map_h*feat_map_w, hidden_dim))
      self.transformer = nn.Transformer(d_model=hidden_dim, nhead=n_heads, num_encoder_layers=n_enc_layers, num_decoder_layers=n_dec_layers, batch_first=True)

      self.linear_param = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid()
      )

  def forward(self, x):
      """
      x: vision feature matrix
      x.shape: (batch_size, feat_map_h*feat_map_w, num_feat_map)
      """
      batch = x.shape[0]
      x = x + self.pos_encoding.unsqueeze(0).contiguous().repeat(batch, 1, 1)   # pos_encoding every pixel on feature map; pos_encoding are replicated for each input in the batch
      x = self.transformer(x, self.query_pos.unsqueeze(0).contiguous().repeat(batch, 1, 1))   # query pos are replicated for each individual event stream sample in the output batch
      x = self.linear_param(x)

      return x # output shape: (batch, num_events, 4); 4: (x,y,p,t)
  
class Transformer(nn.Module):
  def __init__(self, hidden_dim = 64, n_heads=8, n_enc_layers = 3, n_dec_layers=3, max_n_events = 512, feat_map_h = 34, feat_map_w = 34):
      super().__init__()
      """
      check:
      feature map width and height
      """
      self.query_pos = nn.Parameter(torch.rand(max_n_events, hidden_dim)) #to consider: use learnable or fixed?
      self.pos_encoding  = nn.Parameter(torch.rand(feat_map_h*feat_map_w, hidden_dim))
      self.transformer = nn.Transformer(d_model=hidden_dim, nhead=n_heads, num_encoder_layers=n_enc_layers, num_decoder_layers=n_dec_layers, batch_first=True)

  def forward(self, x):
      """
      x: vision feature matrix
      x.shape: (batch_size, feat_map_h*feat_map_w, num_feat_map)
      """
      batch = x.shape[0]
      x = x + self.pos_encoding.unsqueeze(0).contiguous().repeat(batch, 1, 1)   # pos_encoding every pixel on feature map; pos_encoding are replicated for each input in the batch
      x = self.transformer(x, self.query_pos.unsqueeze(0).contiguous().repeat(batch, 1, 1))   # query pos are replicated for each individual event stream sample in the output batch

      return x # output shape: (batch, num_events, 4); 4: (x,y,p,t)
  
class ModelPos(nn.Module):
  def __init__(self, hidden_dim = 64):
      super().__init__()
      """
      check:
      feature map width and height
      """
      self.linear_param = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, 2),
            nn.Sigmoid())

  def forward(self, x):

      x = self.linear_param(x)

      return x # output shape: (batch, num_events, 4); 4: (x,y,p,t)
  
class ModelP(nn.Module):
  def __init__(self, hidden_dim = 64):
      super().__init__()
      """
      check:
      feature map width and height
      """
      self.linear_param = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid())
            

  def forward(self, x):

      x = self.linear_param(x)

      return x # output shape: (batch, num_events, 4); 4: (x,y,p,t)
  
class ModelT(nn.Module):
  def __init__(self, hidden_dim = 64):
      super().__init__()
      """
      check:
      feature map width and height
      """
      self.linear_param = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid())

  def forward(self, x):

      x = self.linear_param(x)

      return x # output shape: (batch, num_events, 4); 4: (x,y,p,t)