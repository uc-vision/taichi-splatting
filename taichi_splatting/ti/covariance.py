import taichi as ti
from taichi.math import vec2, mat2, vec3



@ti.func
def radii_from_cov(uv_cov: mat2) -> ti.f32:
    
    d = (uv_cov[0, 0] - uv_cov[1, 1])
    large_eigen_values = (uv_cov[0, 0] + uv_cov[1, 1] +
                          ti.sqrt(d * d + 4.0 * uv_cov[0, 1] * uv_cov[1, 0])) / 2.0
    # 3.0 is a value from experiment
    return ti.sqrt(large_eigen_values)

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
    conic: vec3,
) -> mat2:
    return mat2([conic.x, conic.y], [conic.y, conic.z]).inverse()



@ti.func
def radii_from_conic(conic: vec3):
    return radii_from_cov(conic_to_cov(conic))



@ti.func
def conic_pdf(xy: vec2, uv: vec2, uv_conic: vec3) -> ti.f32:
    v = xy - uv
    return ti.exp(-0.5 * (v.x * v.x * uv_conic.x + v.y * v.y * uv_conic.z) 
        - v.x * v.y * uv_conic.y)



@ti.func
def cov_inv_basis(uv_cov: mat2, scale: ti.f32) -> mat2:
    basis = ti.Matrix.cols(cov_axes(uv_cov))
    return (basis * scale).inverse()