typedef __BuiltinFloatingPointType Float;



[Differentiable]
T dot<T : Float, let N : int>
  (vector<T, N> v1, vector<T, N> v2) {
  T sum = T(0);

  [ForceUnroll]
  for (int i = 0; i < N; i++) {
    sum +=  v1[i] * v2[i];
  }
  
  return sum;
}

[Differentiable]
vector<T, N> matrix_vector<T : Float, let N:int, let M:int>
  (matrix<T, N, M> mat, vector<T, M> vec) {

  var out = vector<T, N>(0);

  [ForceUnroll]
  for (int i = 0; i < N; i++) {
      out[i] = dot<T, M>(mat[i], vec);
  }
  return out;
}

[Differentiable]
vector<T, N> row<T : Float, let N:int, let M:int>
  (matrix<T, N, M> mat, int row) {
  vector<T, N> out;

  [ForceUnroll]
  for (int i = 0; i < N; i++) {
      out[i] = mat[row][i];
  }
  return out;
}

[Differentiable]
vector<T, M> vector_matrix<T : Float, let N : int, let M : int>
  (vector<T, N> vec, matrix<T, N, M> mat) {

    var out = vector<T, M>(0);

    [ForceUnroll]
    for (int i = 0; i < N; i++) {
        out[i] = dot<T, N>(vec, row<T, N, M>(mat, i));
    }
    return out;
}

// [numthreads(16, 8, 1)]
float add(float a, float b) {
    let g = WorkgroupSize();
    printf("workgroup size: %d", g.x);

    SV_GroupID();
    return a + b;
}

