import taichi as ti
from taichi.math import vec3, mat3, vec4, mat4

@ti.func
def quat_to_mat(q:vec4) -> mat3:
  x, y, z, w = q
  x2, y2, z2 = x*x, y*y, z*z

  return mat3(
    1 - 2*y2 - 2*z2, 2*x*y - 2*w*z, 2*x*z + 2*w*y,
    2*x*y + 2*w*z, 1 - 2*x2 - 2*z2, 2*y*z - 2*w*x,
    2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x2 - 2*y2
  )

@ti.func
def join_rt(r:mat3, t:vec3) -> mat4:
   return mat4(
      r[0, 0], r[0, 1], r[0, 2], t[0],
      r[1, 0], r[1, 1], r[1, 2], t[1],
      r[2, 0], r[2, 1], r[2, 2], t[2],
      0, 0, 0, 1
  )

@ti.func
def qt_to_mat(q:vec4, t:vec3) -> mat4:
  r = quat_to_mat(q)
  return mat4(
    r[0, 0], r[0, 1], r[0, 2], t[0],
    r[1, 0], r[1, 1], r[1, 2], t[1],
    r[2, 0], r[2, 1], r[2, 2], t[2],
    0, 0, 0, 1
  )
  

@ti.func
def scaling_matrix(scale:vec3) -> mat3:
  return mat3(
    scale.x, 0, 0,
    0, scale.y, 0,
    0, 0, scale.z
  )

@ti.func
def quat_mul(q1: ti.math.vec4, q2: ti.math.vec4) -> ti.math.vec4:
    return ti.math.vec4(
        q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y,
        q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x,
        q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w,
        q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z,
    )

@ti.func
def quat_conj(q: ti.math.vec4) -> ti.math.vec4:
    return ti.math.vec4(-q.x, -q.y, -q.z, q.w)


@ti.func
def quat_rotate(q: ti.math.vec4, v: ti.math.vec3) -> ti.math.vec3:
    qv = ti.math.vec4(*v, 0.0)
    q_rot = quat_mul(q, quat_mul(qv, quat_mul(q)))
    return q_rot.xyz

