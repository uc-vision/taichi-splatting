import "matrix.slang";

__generic<T : Float>
typedef vector<T, 3> vec3;



[Differentiable]
T[1] rsh_cart_0<T : Float>(vector<T, 3> xyz) {
  return {T(0.282094791773878)};
}

[Differentiable]
T[4] rsh_cart_1<T : Float>(vector<T, 3> xyz) {
  T x = xyz[0];
  T y = xyz[1];
  T z = xyz[2];
  
  return {
    T(0.282094791773878),
    T(-0.48860251190292) * y,
    T(0.48860251190292) * z,
    T(-0.48860251190292) * x
  };

}

[Differentiable]
T[9] rsh_cart_2<T : Float>(vector<T, 3> xyz) {
  T x = xyz[0];
  T y = xyz[1];
  T z = xyz[2];

  T x2 = x * x;
  T y2 = y * y;
  T z2 = z * z;
  T xy = x * y;
  T xz = x * z;
  T yz = y * z;

  return {
    T(0.282094791773878),
    T(-0.48860251190292) * y,
    T(0.48860251190292) * z,
    T(-0.48860251190292) * x,
    T(1.09254843059208) * xy,
    T(-1.09254843059208) * yz,
    T(0.94617469575756) * z2 - T(0.31539156525252),
    T(-1.09254843059208) * xz,
    T(0.54627421529604) * x2 - T(0.54627421529604) * y2
  };

}

[Differentiable]
T[16] rsh_cart_3<T : Float>(vector<T, 3> xyz) {
    T x = xyz[0];
    T y = xyz[1];
    T z = xyz[2];

    T x2 = x * x;
    T y2 = y * y;
    T z2 = z * z;
    T xy = x * y;
    T xz = x * z;
    T yz = y * z;

    return {
        T(0.282094791773878),
        T(-0.48860251190292) * y,
        T(0.48860251190292) * z,
        T(-0.48860251190292) * x,
        T(1.09254843059208) * xy,
        T(-1.09254843059208) * yz,
        T(0.94617469575756) * z2 - T(0.31539156525252),
        T(-1.09254843059208) * xz,
        T(0.54627421529604) * x2 - T(0.54627421529604) * y2,
        T(-0.590043589926644) * y * (T(3.0) * x2 - y2),
        T(2.89061144264055) * xy * z,
        T(0.304697199642977) * y * (T(1.5) - T(7.5) * z2),
        T(1.24392110863372) * z * (T(1.5) * z2 - T(0.5)) - T(0.497568443453487) * z,
        T(0.304697199642977) * x * (T(1.5) - T(7.5) * z2),
        T(1.44530572132028) * z * (x2 - y2),
        T(-0.590043589926644) * x * (x2 - T(3.0) * y2)
    };

}
[Differentiable]
vector<T, 3> as_vector<T : Float>(T[3] data) {
  return {data[0], data[1], data[2]};
}


[Differentiable]
vec3<T> evaluate_sh0<T : Float>(T[1][3] sh_params, vec3<T> position, vec3<T> camera_pos) {
    T[1] coeffs = rsh_cart_0<T>(normalize(position - camera_pos));
    return as_vector<T>(vector_matrix<T, 1, 3>(coeffs, sh_params));
}


[Differentiable]
vec3<T> evaluate_sh1<T : Float>(T[4][3] sh_params, vec3<T> position, vec3<T> camera_pos) {
    T[4] coeffs = rsh_cart_1<T>(normalize(position - camera_pos));
    return as_vector<T>(vector_matrix<T, 4, 3>(coeffs, sh_params));
}

[Differentiable]
vec3<T> evaluate_sh2<T : Float>(T[9][3] sh_params, vec3<T> position, vec3<T> camera_pos) {
    T[9] coeffs = rsh_cart_2<T>(normalize(position - camera_pos));
    return as_vector<T>(vector_matrix<T, 9, 3>(coeffs, sh_params));
}

[Differentiable]
vec3<T> evaluate_sh3<T : Float>(T[16][3] sh_params, vec3<T> position, vec3<T> camera_pos) {
    T[16] coeffs = rsh_cart_3<T>(normalize(position - camera_pos));
    let c = vector_matrix<T, 16, 3>(coeffs, sh_params);

    return as_vector<T>(c);
}
