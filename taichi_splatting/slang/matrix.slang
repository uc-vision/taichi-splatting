typedef __BuiltinFloatingPointType Float;



[Differentiable]
T dot<T : Float, let N : int>
  (T[N] v1, T[N] v2) {
    T sum = T(0);

    [ForceUnroll]
    for (int i = 0; i < N; i++) {
        sum += v1[i] * v2[i];
    }

    return sum;
}

[Differentiable]
T[N] matrix_vector<T : Float, let N:int, let M:int>
  (T[N][M] mat, T[M] vec) {

  T[N] out;

  [ForceUnroll]
  for (int i = 0; i < N; i++) {
    let v = row<T, N, M>(mat, i);
    out[i] = dot<T, M>(v, vec);
  }

  return out;
}

[Differentiable]
T[M] vector_matrix<T : Float, let N : int, let M : int>
  (T[N] vec, T[N][M] mat) {

    T[M] out;

    [ForceUnroll]
    for (int i = 0; i < M; i++) {
        let v = mat[i];
        out[i] = dot<T, N>(v, vec);
    }

    return out;
}

[Differentiable]
T[M] row<T : Float, let N:int, let M:int>
  (T[N][M] mat, int row) {
  T[M] out;

  [ForceUnroll]
  for (int i = 0; i < M; i++) {
  
      out[i] = mat[row][i];
  }
  return out;
}


