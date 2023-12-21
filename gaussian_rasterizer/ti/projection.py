import taichi as ti
from taichi.math import vec2, vec3, vec4, mat2, mat3, mat4
from gaussian_rasterizer.ti.transforms import quat_to_mat

from gaussian_rasterizer.torch.transforms import split_rt
mat2x3f = ti.types.matrix(n=2, m=3, dtype=ti.f32)

@ti.func
def point_to_camera(
    position: vec3,
    T_camera_world: mat4,
    projective_transform: mat3,
):
    point_in_camera = (T_camera_world @ vec4(*position, 1)).xyz
    uv = (projective_transform @ point_in_camera) / point_in_camera.z
    return uv.xy, point_in_camera


def camera_origin(T_camera_world: mat4):
  r, t = split_rt(T_camera_world)
  t = -(r.transpose() @ t)
  return t



@ti.func
def get_projective_transform_jacobian(
    projection: mat3,
    position: vec3,
):
    # cx = projective_transform[0, 2]
    # cy = projective_transform[1, 2]
    # [[fx/z, 0, cx/z - (cx*z + fx*x)/z**2], [0, fy/z, cy/z - (cy*z + fy*y)/z**2]]
    fx = projection[0, 0]
    fy = projection[1, 1]

    x, y, z = position
    return mat2x3f([
        [fx / z, 0, -(fx * x) / (z * z)],
        [0, fy / z, -(fy * y) / (z * z)]
    ])


@ti.func
def gaussian_covariance_in_camera(
    T_camera_world: mat4,
    cov_rotation: vec4,
    cov_scale: vec3,
) -> mat3:
    """ Construct and rotate the covariance matrix in camera space
    """
    
    W = T_camera_world[:3, :3]
    R = quat_to_mat(cov_rotation)

    S = mat3([
        [cov_scale.x, 0, 0],
        [0, cov_scale.y, 0],
        [0, 0, cov_scale.z]
    ])
    # covariance matrix, 3x3, equation (6) in the paper
    # Sigma = R @ S @ S.transpose() @ R.transpose()
    # cov_uv = J @ W @ Sigma @ W.transpose() @ J.transpose()  # equation (5) in the paper
    
    m = W @ R @ S
    return m @ m.transpose() 

@ti.func
def project_gaussian_to_image(
    projective_transform: mat3,
    point_in_camera: vec3,
    cov_in_camera: mat3) -> mat2:

    """ Approximate the 2D gaussian covariance in image space """

    J = get_projective_transform_jacobian(
        projective_transform, point_in_camera)
    
    
    cov_uv = J @ cov_in_camera @ J.transpose()
    return cov_uv

