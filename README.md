# Taichi Splatting

Rasterizer for Guassian Splatting using Taichi and PyTorch - embedded in python library. 

## Progress

### Done
* Simple view culling 
* Projection (no gradient yet)
* Tile mapping 
* Rasterizer forward pass
* Spherical harmonics with autograd

### Todo
* Rasterizer backward pass
* Projection autograd wrapper

### Modifications

* Exposed all internal constants as parameters
* Switched to matrices as inputs instead of quaternions 
* Tile mapping tighter culling for tile overlaps (~30% less rendered splats!)


## Attribution - Taichi 3D Guassian Splatting

This work is largely derived off Taichi 3D Gaussian Splatting: https://github.com/wanmeihuali/taichi_3d_gaussian_splatting

Key differences are the rendering algorithm is decomposed into separate operations (projection, shading functions, tile mapping and rasterization) which can be combined in different ways in order to facilitate a more flexible use. Using the Taichi autodiff for a simpeler implementation. 

For example projecting features for label transfer, colours via. spherical harmonics or depth with depth covariance without needing to build it into the renderer and remaining differentiable.
