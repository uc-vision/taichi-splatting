from typing import List, Tuple
import torch
import torch.nn as nn

from taichi_splatting.misc.renderer2d import point_covariance
from torch.nn import TransformerEncoder, TransformerEncoderLayer, MultiheadAttention
from typing import List
from .renderer2d import Gaussians2D

import torch.nn.functional as F

def kl_divergence(means1:torch.Tensor, means2:torch.Tensor, cov1:torch.Tensor, cov2:torch.Tensor) -> torch.Tensor:
  """Compute KL divergence between two 2D Gaussian distributions.
  
  Args:
    means1: (N,D) tensor of means for first distribution
    means2: (N,D) tensor of means for second distribution  
    cov1: (N,D,D) tensor of covariance matrices for first distribution
    cov2: (N,D,D) tensor of covariance matrices for second distribution

  Returns:
    (N,) tensor of KL divergence values
  """
  # Compute inverse of cov2
  cov2_inv = torch.linalg.inv(cov2)
  
  # Compute trace term
  trace_term = torch.einsum('...ij,...jk->...ik', cov2_inv, cov1)
  trace_term = torch.diagonal(trace_term, dim1=-2, dim2=-1).sum(-1)
  
  # Compute quadratic term for means
  mean_diff = means2 - means1
  quad_term = torch.einsum('...i,...ij,...j->...', mean_diff, cov2_inv, mean_diff)
  
  # Combine terms
  return 0.5 * (trace_term + quad_term - 2 + torch.logdet(cov2) - torch.logdet(cov1))
  


class InputResidual(nn.Module):
  def __init__(self, *layers):
    super().__init__()
    self.layers = nn.ModuleList(layers)

  def forward(self, inputs):
    x = self.layers[0](inputs)
    
    for layer in self.layers[1:]:
      x = layer(torch.cat([x, inputs], dim=1))
    return x

def linear(in_features, out_features,  init_std=None):
  m = nn.Linear(in_features, out_features, bias=True)

  if init_std is not None:
    m.weight.data.normal_(0, init_std)
    
  m.bias.data.zero_()
  return m

class Residual(nn.Module):
  def __init__(self, *layers):
    super().__init__()
    self.layers = nn.ModuleList(layers)

  def forward(self, x):
    for layer in self.layers:
      x = x + layer(x)
    return x


def layer(in_features, out_features, activation=nn.ReLU, norm=nn.Identity, **kwargs):
  return nn.Sequential(linear(in_features, out_features, **kwargs), 
                       activation(),
                       norm(out_features),
                       nn.Dropout(0.4)
                       )


def mlp(inputs, outputs, hidden_channels: List[int], activation=nn.ReLU, norm=nn.Identity, 
        output_activation=nn.Identity, output_scale =None):

  output_layer = layer(hidden_channels[-1], outputs, 
                       output_activation,
                       init_std=output_scale)
  
  return nn.Sequential(
    layer(inputs, hidden_channels[0], activation),
    *[layer(hidden_channels[i], hidden_channels[i+1], activation, norm)  
      for i in range(len(hidden_channels) - 1)],
    output_layer,
  )   
class SelfAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout_prob=0.1):
        super(SelfAttentionLayer, self).__init__()
        # Multi-Head Self Attention
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout_prob, batch_first=True)
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
        # Dropout
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        # Apply multi-head self-attention
        attn_output, _ = self.attention(x, x, x)
        # Add residual connection and apply layer normalization
        x = self.layer_norm(x + self.dropout(attn_output))
        return x

class TransformerMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_channels: list, 
                 num_heads: int = 4, num_layers: int = 4, activation=nn.ReLU, 
                 dropout_prob: float = 0.1, max_positions: int = 1024, dims: int = 2):
        super(TransformerMLP, self).__init__()
        
        self.dims = dims  # Number of spatial dimensions for Z-order curve

        # Z-Order Positional Encoding
        self.positional_embedding = nn.Embedding(max_positions, input_dim)
        
        # Transformer Encoder Layer
        encoder_layer = TransformerEncoderLayer(
            d_model=input_dim,   # The number of expected features in the input
            nhead=num_heads,     # Number of attention heads
            dim_feedforward=hidden_channels[0],  # Feedforward hidden size
            activation=activation(),  # Activation function
            dropout=dropout_prob  # Dropout for transformer encoder layer
        )
        
        # Create a Transformer Encoder
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Multi-Head Self Attention Layer
        self.attention = MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, dropout=dropout_prob)
        
        # Feed-Forward Layers with Dropout
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_channels[0]),
            activation(),
            nn.Dropout(dropout_prob),  # Dropout Layer
            nn.Linear(hidden_channels[0], output_dim),
        )
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(input_dim)
        
        # Residual Connection with Dropout
        self.residual = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            activation(),
            nn.Dropout(dropout_prob)  # Dropout for residual connection
        )

    def compute_default_positions(self, batch_size, seq_len):
        """
        Compute default 2D grid positions for Z-order encoding.
        :param batch_size: Number of samples in the batch
        :param seq_len: Length of the sequence
        :return: Default grid positions of shape (batch_size, seq_len, dims)
        """
        grid_positions = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
        positions = torch.stack([grid_positions // int(seq_len**0.5), grid_positions % int(seq_len**0.5)], dim=-1)
        return positions.to(next(self.positional_embedding.parameters()).device)

    def compute_z_order_indices(self, positions):
        """
        Compute simple Z-order indices for input positions.
        :param positions: Tensor of shape (batch_size, seq_len, dims)
        :return: Tensor of Z-order indices
        """
        z_order = torch.zeros(positions.size(0), positions.size(1), dtype=torch.long, device=positions.device)
        for i in range(self.dims):
            z_order |= (positions[:, :, i] << i)  # Bitwise interleave
        return z_order

    def forward(self, x, positions=None):
      """
      :param x: Input tensor of shape (seq_len, batch_size, input_dim)
      :param positions: Input positions for Z-order encoding (batch_size, seq_len, dims)
      """
      if positions is None:
          # Compute default positions if not provided
          positions = self.compute_default_positions(batch_size=x.size(1), seq_len=x.size(0))

      # Compute Z-Order Indices
      z_order_indices = self.compute_z_order_indices(positions)
      
      # Embed Positional Encodings
      z_order_embeddings = self.positional_embedding(z_order_indices)  # Shape: (batch_size, seq_len, input_dim)
      
      # Permute embeddings to match input shape (seq_len, batch_size, input_dim)
      z_order_embeddings = z_order_embeddings.permute(1, 0, 2)
      
      # Add Positional Encodings to Input
      x = x + z_order_embeddings

      # Transformer Encoder Pass
      transformer_output = self.transformer_encoder(x)
      
      # Residual Connection (Skip Connection)
      x = x + transformer_output
      
      # Layer Normalization
      x = self.layer_norm(x)

      # Apply Multi-Head Attention
      attn_output, _ = self.attention(x, x, x)
      
      # Residual Connection with Attention Output
      x = x + attn_output
      
      # Apply Feed-Forward Network
      output = self.ffn(x)
      
      return output


class AttentionGate(nn.Module):
    def __init__(self, in_channels, gate_channels, inter_channels):
        super(AttentionGate, self).__init__()
        self.theta_x = nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0)
        self.phi_g = nn.Conv2d(gate_channels, inter_channels, kernel_size=1, stride=1, padding=0)
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        # Apply transformations
        theta_x = self.theta_x(x)
        phi_g = self.phi_g(g)
        add_xg = self.relu(theta_x + phi_g)
        psi = self.sigmoid(self.psi(add_xg))
        # Upsample psi to match the spatial dimensions of x
        psi = F.interpolate(psi, size=x.shape[2:], mode="bilinear", align_corners=True)
        # Apply attention to x
        return x * psi

class UNet4(nn.Module):
    def __init__(self, dropout_rate=0.5, l2_lambda=0.01):
        super(UNet4, self).__init__()
        
        # Regularizer (L2 is typically handled by the optimizer in PyTorch)
        self.dropout_rate = dropout_rate

        # Contracting Path
        self.enc1 = self.conv_block(1, 64, dropout_rate)
        self.enc2 = self.conv_block(64, 128, dropout_rate)
        self.enc3 = self.conv_block(128, 256, dropout_rate)
        self.bottleneck = self.conv_block(256, 512, dropout_rate)

        # Expanding Path
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.attention3 = AttentionGate(256, 256, inter_channels=128)
        self.dec3 = self.conv_block(512, 256, dropout_rate)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.attention2 = AttentionGate(128, 128, inter_channels=64)
        self.dec2 = self.conv_block(256, 128, dropout_rate)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.attention1 = AttentionGate(64, 64, inter_channels=32)
        self.dec1 = self.conv_block(128, 64, dropout_rate)

        # Output Layer
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

    def conv_block(self, in_channels, out_channels, dropout_rate):
        """
        Creates a convolutional block with Conv2D -> BatchNorm -> ReLU -> Dropout.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )

    def forward(self, x):
        # Contracting Path
        x1 = self.enc1(x)
        p1 = F.max_pool2d(x1, kernel_size=2)

        x2 = self.enc2(p1)
        p2 = F.max_pool2d(x2, kernel_size=2)

        x3 = self.enc3(p2)
        p3 = F.max_pool2d(x3, kernel_size=2)

        # Bottleneck
        x4 = self.bottleneck(p3)

        # Expanding Path
        up3 = self.upconv3(x4)
        print(x3.shape)
        print(up3.shape)
        att3 = self.attention3(x3, up3)
        concat3 = torch.cat([up3, att3], dim=1)
        x5 = self.dec3(concat3)

        up2 = self.upconv2(x5)
        print(x5.shape)
        print(up2.shape)
        att2 = self.attention2(x2, up2)
        concat2 = torch.cat([up2, att2], dim=1)
        x6 = self.dec2(concat2)

        up1 = self.upconv1(x6)
        att1 = self.attention1(x1, up1)
        concat1 = torch.cat([up1, att1], dim=1)
        x7 = self.dec1(concat1)

        # Output Layer
        output = self.output_layer(x7)

        return output