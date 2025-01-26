from functools import partial
import math
from taichi_splatting.data_types import Gaussians2D, inverse_sigmoid
from taichi_splatting.tests.random_data import random_2d_gaussians
import torch.nn as nn
from typing import Iterator, List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

def linear(in_features, out_features,  init_std=None):
  m = nn.Linear(in_features, out_features, bias=True)

  if init_std is not None:
    m.weight.data.normal_(0, init_std)
    
  m.bias.data.zero_()
  return m


def layer(in_features, out_features, activation=nn.Identity, norm=nn.Identity, **kwargs):
  return nn.Sequential(linear(in_features, out_features, **kwargs), 
                       norm(out_features),
                       activation(),
                       )


def mlp_body(inputs, hidden_channels: List[int], activation=nn.ReLU, norm=nn.Identity):
  return nn.Sequential(
    layer(inputs, hidden_channels[0], activation),
    *[layer(hidden_channels[i], hidden_channels[i+1], activation, norm)  
      for i in range(len(hidden_channels) - 1)]
  )   


def mlp(inputs, outputs, hidden_channels: List[int], activation=nn.ReLU, norm=nn.Identity, 
        output_activation=nn.Identity, output_scale =None):

  output_layer = layer(hidden_channels[-1], outputs, 
                       output_activation,
                       init_std=output_scale)
  
  return nn.Sequential(
    mlp_body(inputs, hidden_channels, activation, norm),
    output_layer
  )   


class Autoencoder(nn.Module):
  def __init__(self, encoder, decoder):
    super().__init__()
    self.encoder = encoder
    self.decoder = decoder

  def forward(self, x):
    latent = self.encode(x)
    return latent, self.decoder(latent)
  
  def encode(self, x):
    latent = self.encoder(x)
    return latent


class SinCos(nn.Module):
  def __init__(self):
    super(SinCos, self).__init__()

  def forward(self, x):
    return torch.cat([torch.sin(x), torch.cos(x)], dim=1)

def sincos(x):
  return torch.cat([torch.sin(x), torch.cos(x)], dim=1)





class RandomProjections(nn.Module):
  """
  "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
      https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html
  """

  def __init__(self, n_projections:int, dims:int, scale_range:tuple[float, float], trainable=False):
    super().__init__()
    
    scales = torch.linspace(
        math.log(scale_range[0]), math.log(scale_range[1]), n_projections
    )
    self.scales = nn.Parameter(scales, requires_grad=trainable)

    self.planes = nn.Parameter(torch.randn(n_projections, dims) * 2 * torch.pi, requires_grad=trainable)
    self.offset = nn.Parameter(torch.randn(n_projections) * 2 * torch.pi, requires_grad=False)

  def forward(self, x):
    planes = F.normalize(self.planes, dim=1) / self.scales.exp().unsqueeze(1)
    return sincos(x @ planes.T + self.offset)
  
  
class RandomUnprojections(nn.Module):
  def __init__(self, n_projections:int, dims:int, scale_range:tuple[float, float], trainable=False):
    super().__init__()
    
    scales = torch.linspace(
        math.log(scale_range[0]), math.log(scale_range[1]), n_projections
    )
    # scales = torch.linspace(
    #     math.log(scale_range[0]), math.log(scale_range[1]), n_projections
    # ).exp()

    self.scales = nn.Parameter(scales, requires_grad=False)
    
    self.planes = nn.Parameter(
        torch.randn(n_projections, dims), 
        requires_grad=trainable
    )
    

  def forward(self, features:torch.Tensor) -> torch.Tensor:
    n_proj, n_dim = self.planes.shape
    assert features.shape[1] // n_dim == n_proj, f"Wrong feature shape: {features.shape} expected {(n_proj, n_dim)}"


    coords = features.view(-1, n_dim, n_proj)  # [n_projections, 2]
    return (coords * self.scales.exp().unsqueeze(0)).sum(dim=2)
  


class PositionAutoencoder(nn.Module):
  def __init__(self, encoder:nn.Module, decoder:nn.Module, n_projections:int=32, dims:int=2, trainable_projections=False, scale_range=(1.0, 2048.0)):
    super().__init__()
    self.encoder = encoder
    self.decoder = decoder

    self.project = RandomProjections(n_projections, dims, scale_range, trainable_projections)
    self.unproject = RandomUnprojections(n_projections, dims, scale_range, trainable_projections)
    
  def forward(self, x:torch.Tensor                    # [n_batch, n_dims]
              ) -> Tuple[torch.Tensor, torch.Tensor]: # [n_batch, latent_dims], [n_batch, n_dims]
    
    latent = self.encoder(self.project(x))

    decoded_features = self.decoder(latent)
    return latent, self.unproject(decoded_features)


def random_gaussians(n, image_size_range:tuple[int, int], 
                        num_channels=3, 
                        alpha_range=(0.001, 0.999), 
                        depth_range=(0.0, 1.0),
                        device=torch.device('cuda', 0)
                      ) -> Gaussians2D:
  w = torch.randint(image_size_range[0], (n,))
  h = w * (0.5 + torch.rand(n, device=device))  # This gives height between 0.5w and 1.5w

  position = torch.rand(n, 2, device=device) * torch.tensor([w, h], dtype=torch.float32, device=device).unsqueeze(0)
  depth = torch.rand((n, 1), device=device) * (depth_range[1] - depth_range[0]) + depth_range[0]
  
  mean_scale = w / (1 + math.sqrt(n))
  log_scaling = torch.randn(n, 2, device=device) + math.log(mean_scale)

  rotation = F.normalize(torch.randn(n, 2, device=device), dim=1)

  low, high = alpha_range
  alpha = torch.rand(n, device=device) * (high - low) + low

  return Gaussians2D(
    position=position,
    depths=depth,
    log_scaling=log_scaling,
    rotation=rotation,
    alpha_logit=inverse_sigmoid(alpha),
    feature=torch.rand(n, num_channels, device=device),
    batch_size=(n,)
  )


def mlp_pair(inputs, outputs, hidden_channels: List[int], activation=nn.ReLU, norm=nn.Identity):
  encoder = mlp(inputs, outputs, hidden_channels, activation, norm)
  decoder = mlp(outputs, inputs, list(reversed(hidden_channels)), activation, norm)
  return encoder, decoder

def position_autoencoder(dims:int, n_projections:int, latent_dims:int, 
                        hidden_channels: List[int], activation=nn.ReLU, norm=nn.Identity,
                        scale_range=(0.1, 2048.0), trainable_projections=True):
  projected_size = n_projections * 2

  encoder, decoder = mlp_pair(projected_size, latent_dims, hidden_channels, activation, norm)
  return PositionAutoencoder(encoder, decoder, n_projections, dims, scale_range=scale_range, trainable_projections=trainable_projections)


def generate_gaussians(image_size_range:tuple[int, int], num_gaussians:int, device=torch.device):
  while True:
    gaussians = random_gaussians(num_gaussians, image_size_range, alpha_range=(0.5, 1.0), scale_factor=0.5, device=device)
    yield gaussians



def generate_points(n_points:int, image_size_range:tuple[int, int], device=torch.device):
  while True:
    w = torch.randint(image_size_range[0], (n_points,), device=device)
    h = w * (0.5 + torch.rand(n_points, device=device))  # This gives height between 0.5w and 1.5w

    norm_points = torch.rand(n_points, 2, device=device) 
    size = torch.stack([w, h], dim=1)

    yield norm_points * size - size / 2



def train_position_autoencoder(n_iter:int, iter_points:Iterator, device=torch.device, noise_level:float=10.00):
  autoencoder = position_autoencoder(2, n_projections=32, latent_dims=64, hidden_channels=[64], 
                                     activation=nn.SiLU, 
                                     scale_range=(1e-4, 1e4), trainable_projections=True)
  autoencoder.to(device)

  autoencoder = torch.compile(autoencoder)

  optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3, betas=(0.9, 0.999))
  scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99995)

  pbar = tqdm(desc="Training", total=n_iter)
  for i in range(n_iter):

    points = next(iter_points).to(device)
    optimizer.zero_grad()
    latent, decoded = autoencoder(points + torch.randn_like(points) * noise_level)
    
    # Calculate MSE loss first since we display it
    loss = F.mse_loss(decoded, points) 
    
    loss.backward()
    optimizer.step()

    if i % 100 == 0:

      points = next(iter_points).to(device)

      latent, decoded = autoencoder(points)
      mse = F.mse_loss(decoded, points, reduction='none')

      pbar.update(100)
      pbar.set_postfix_str(f"loss={loss.item():<10.4f} mse={mse.mean().item():<10.4f} worst_l1={mse.max().sqrt().item():>10.3f} lr={optimizer.param_groups[0]['lr']:<8.4e}")


    # Anneal learning rate using exponential decay
    scheduler.step()
    noise_level *= 0.9999

    
  pbar.close()




if __name__ == "__main__":
  torch.set_float32_matmul_precision('high')

  device = torch.device('cuda', 0)  
  iter = generate_points(2**14, (512, 2048), device=device)
  train_position_autoencoder(n_iter = 100000, iter_points=iter, device=device)
