import taichi as ti
import numpy as np

from taichi.math import vec2, mat2, vec3



@ti.func
def radii_from_cov(uv_cov: mat2) -> ti.f32:
    
    d = (uv_cov[0, 0] - uv_cov[1, 1])
    large_eigen_values = (uv_cov[0, 0] + uv_cov[1, 1] +
                          ti.sqrt(d * d + 4.0 * uv_cov[0, 1] * uv_cov[1, 0])) / 2.0
    # 3.0 is a value from experiment
    return ti.sqrt(large_eigen_values) * 3.0

@ti.func
def cov_axes(A):
    tr = A.trace()
    det = A.determinant()
    sqrt_gap = ti.sqrt(tr**2 - 4 * det)

    lambda1 = (tr + sqrt_gap) * 0.5
    lambda2 = (tr - sqrt_gap) * 0.5

    v1 = vec2(A[0, 0] - lambda2, A[1, 0]).normalized()
    v2 = vec2(A[0, 0] - lambda1, A[1, 0]).normalized()
    
    return v1 * ti.sqrt(lambda1), v2 * ti.sqrt(lambda2)  


@ti.func
def cov_to_conic(
    gaussian_covariance: mat2,
) -> vec3:
    inv_cov = gaussian_covariance.inverse()
    return vec3(inv_cov[0, 0], inv_cov[0, 1], inv_cov[1, 1])

@ti.func
def conic_to_cov(
    conic: ti.math.vec3,
) -> mat2:
    return mat2([conic.x, conic.y], [conic.y, conic.z]).inverse()



@ti.func
def radii_from_conic(conic: ti.math.vec3):
    return radii_from_cov(conic_to_cov(conic))

