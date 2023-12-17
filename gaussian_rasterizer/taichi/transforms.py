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
def mat_to_quat(m:mat3) -> vec4:

  m00, m01, m02 = m[0]
  m10, m11, m12 = m[1]
  m20, m21, m22 = m[2]

  t = m00 + m11 + m22

  if t > 0:
    s = 0.5 / ti.sqrt(t + 1)
    w = 0.25 / s
    x = (m21 - m12) * s
    y = (m02 - m20) * s
    z = (m10 - m01) * s
  elif m00 > m11 and m00 > m22:
    s = 2 * ti.sqrt(1 + m00 - m11 - m22)
    w = (m21 - m12) / s
    x = 0.25 * s
    y = (m01 + m10) / s
    z = (m02 + m20) / s
  elif m11 > m22:
    s = 2 * ti.sqrt(1 + m11 - m00 - m22)
    w = (m02 - m20) / s
    x = (m01 + m10) / s
    y = 0.25 * s
    z = (m12 + m21) / s
  else:
    s = 2 * ti.sqrt(1 + m22 - m00 - m11)
    w = (m10 - m01) / s
    x = (m02 + m20) / s
    y = (m12 + m21) / s
    z = 0.25 * s

  return vec4(w, x, y, z)

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