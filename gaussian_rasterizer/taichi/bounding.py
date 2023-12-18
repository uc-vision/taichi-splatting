import taichi as ti
from taichi.math import mat2, vec2

mat4x2f = ti.types.matrix(4, 2, dtype=ti.f32)


@ti.dataclass
class OBB:
  centre : vec2  
  corners : mat4x2f
  basis: mat2

  @ti.func
  def relative_point(self, p):
    r = (p - self.centre)
    return self.basis @ r
  
  @ti.func
  def separates(self, points:ti.template()) -> ti.u1:
    separates = False

    rel_points = ti.Matrix.rows([points[:, 0] - self.centre[0], points[:, 1] - self.centre[1]])
    local_points = (self.basis  @ rel_points).transpose()

    for i in ti.static(range(2)):
      min_val = ti.min(*local_points[:, i])
      max_val = ti.max(*local_points[:, i])
      if (min_val > 1. or max_val < -1.):
        separates = True
        
    return separates
  
  @ti.func
  def all(v : ti.template()):
    cond = False
    for i in ti.static(range(v.n)):
      cond = cond and v[i]

    return cond
  
  @ti.func
  def inside(self, point:ti.math.vec2) -> ti.u1:
    rel_point = point - self.centre
    local_point = self.basis @ rel_point

    return all(local_point <= 1.) and all(local_point >= -1.)


@ti.func
def create_obb(centre:vec2, dir:vec2, scales:vec2) -> OBB:
  dir2 = vec2(-dir[1], dir[0]) 

  v1 = dir * scales[0]
  v2 = dir2 * scales[1]

  basis = ti.Matrix.cols([v1, v2]).inverse()

  corners = ti.Matrix.rows([
    centre + v1 + v2,
    centre - v1 + v2,
    centre - v1 - v2,
    centre + v1 - v2
  ])
  
  return OBB(centre, corners, basis)


@ti.dataclass
class AABB:
  lower : vec2
  upper : vec2  


  @ti.func
  def corners(self) -> mat4x2f:
    return ti.Matrix.rows([
      self.upper,
      vec2(self.lower[0], self.upper[1]),
      self.lower,
      vec2(self.upper[0], self.lower[1])
    ])
  
  @ti.func
  def contains(self, p:vec2) -> ti.u1:
    return ti.all(p <= self.upper) and ti.all(p >= self.lower)
  
  @ti.func
  def separates(self, corners:mat4x2f) -> ti.u1:
    separates = False
    for i in ti.static(range(2)):
      min_val = ti.min(*corners[:, i])
      max_val = ti.max(*corners[:, i])
      if min_val > self.upper[i] or max_val < self.lower[i]:
        separates = True
      
    return separates
  