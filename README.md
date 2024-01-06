# Taichi Splatting

Rasterizer for Guassian Splatting using Taichi and PyTorch - embedded in python library. 

This work is originally derived off [Taichi 3D Gaussian Splatting](https://github.com/wanmeihuali/taichi_3d_gaussian_splatting), with significant re-organisation and changes.

Key differences are the rendering algorithm is decomposed into separate operations (projection, shading functions, tile mapping and rasterization) which can be combined in different ways in order to facilitate a more flexible use, and gradients can be enabled on "all the things" as required for the application (and not when disabled, to save performance).

Using the Taichi autodiff for a simpler implementation where possible (e.g. for projection, but not for the rasterization).

Examples:
  * Projecting features for lifting 2D to 3D
  * Colours via. spherical harmonics
  * Depth covariance without needing to build it into the renderer and remaining differentiable.
  * Fully differentiable camera parameters (and ability to swap in new camera models)

## Major dependencies

* taichi-nightly

Some bug fixes which have occurred after the 1.7.0 release:
`pip install --upgrade -i https://pypi.taichi.graphics/simple/ taichi-nightly`

* torch >= 1.8 (probably works with earlier versions, too)

## Installing

* Install taichi-nightly as above (note that the external repo cannot be listed in the pyproject.toml)
* Clone down with `git clone` and install with `pip install ./taichi-spatting`
* `pip install taichi-splatting`


## Executables

### fit_image_gaussians

There exists a toy optimizer for fitting a set of randomly initialized gaussians to some 2D images `fit_image_gaussians` - useful for testing rasterization without the rest of the dependencies.

### benchmarks

There exist benchmarks to evaluate performance on individual components in isolation under `taichi_splatting/benchmarks/`

### tests 

Tests (gradient tests and tests comparing to torch-based reference implementations) can be run with pytest, or individually under 
`taichi_splatting/tests/`

### splat-viewer

A viewer for reconstructions created with the original gaussian-splatting repository can be found [here](https://github.com/uc-vision/splat-viewer) or installed with pip. Has dependencies on open3d and Qt. 

### splat-benchmark

A benchmark for a full rendererer (in the same repository as above) with real reconstructions (rendering the original camera viewpoints).  Options exist for tweaking all the renderer parameters, benchmarking backward pass etc.


## Progress

### Done
* Simple view culling 
* Projection with autograd
* Tile mapping 
* Rasterizer forward pass
* Spherical harmonics with autograd
* Gradient tests for most parts (float64)
* Fit to image training example/test
* Depth and depth-covariance rendering

### Todo

* Depth covariance example
* Benchmark with original + taichi_3dgs rasterizer

* 3D training code (likely different repository)
* Backward projection autograd takes a while to compile and is not cached properly

### Improvements

* Exposed all internal constants as parameters
* Switched to matrices as inputs instead of quaternions
* Tile mapping tighter culling for tile overlaps (~30% less rendered splats!)
* All configuration parameters exposed (e.g. tile_size, saturation threshold etc.)
* Warp reduction based backward pass for rasterizer, a decent boost in performance


## Conventions

### Transformation matrices

Transformations are notated `T_x_y`, for example `T_camera_world` can be used to transform points in the world to points in the local camera by `points_camera = T_camera_world @ points_world`

