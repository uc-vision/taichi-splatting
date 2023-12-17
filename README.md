### Taichi Gaussian Rasterizer

Rasterizer for Guassian Splatting using Taichi and PyTorch - embedded in python library. 

Based off Taichi 3D Gaussian Splatting: https://github.com/wanmeihuali/taichi_3d_gaussian_splatting
Key differences are the rendering is decomposed into projection and rasterization in order to facilitate a more flexible use. Using the Taichi autodiff for a simpeler implementation. 
For example projecting features for label transfer, colours via. spherical harmonics or depth with depth covariance without needing to build it into the renderer and remaining differentiable.
