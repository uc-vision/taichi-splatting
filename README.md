# Taichi Gaussian Rasterizer

Rasterizer for Guassian Splatting using Taichi and PyTorch - embedded in python library. 

## Differences and similarities with Taichi 3D Guassian Splatting

This work is largely derived off Taichi 3D Gaussian Splatting: https://github.com/wanmeihuali/taichi_3d_gaussian_splatting

Key differences are the rendering algorithm is decomposed into separate operations (projection, shading functions, tile mapping and rasterization) which can be combined in different ways in order to facilitate a more flexible use. Using the Taichi autodiff for a simpeler implementation. 

For example projecting features for label transfer, colours via. spherical harmonics or depth with depth covariance without needing to build it into the renderer and remaining differentiable.
