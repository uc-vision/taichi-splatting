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

    gap = tr**2 - 4 * det
    sqrt_gap = ti.sqrt(ti.max(gap, 0))

    lambda1 = (tr + sqrt_gap) * 0.5
    lambda2 = (tr - sqrt_gap) * 0.5

    v1 = vec2(A[0, 0] - lambda2, A[1, 0]).normalized() 
    v2 = vec2(v1.y, -v1.x)

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
    dx, dy = xy - uv
    a, b, c = uv_conic

    p = ti.exp(-0.5 * (dx**2 * a + dy**2 * c) - dx * dy * b)
    return p


@ti.func
def conic_pdf_with_grad(xy: vec2, uv: vec2, uv_conic: vec3):
    d = xy - uv
    a, b, c = uv_conic

    dx2 = d.x**2
    dy2 = d.y**2
    dxdy = d.x * d.y
    
    p = ti.exp(-0.5 * (dx2 * a + dy2 * c) - dxdy * b)
    dp_duv = vec2(
        (b * d.y - 0.5 * a * (2 * uv.x - 2 * xy.x)) * p,
        (b * d.x - 0.5 * c * (2 * uv.y - 2 * xy.y)) * p
    )
    dp_dconic = vec3(-0.5 * dx2 * p, -dxdy * p, -0.5 * dy2 * p)

    return p, dp_duv, dp_dconic


@ti.func
def cov_inv_basis(uv_cov: mat2, scale: ti.f32) -> mat2:
    basis = ti.Matrix.cols(cov_axes(uv_cov))
    return (basis * scale).inverse()

@ti.func
def isfin(x):
  return ~(ti.math.isinf(x) or ti.math.isnan(x))