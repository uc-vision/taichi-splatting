import taichi as ti


@ti.kernel
def split_kernel(indexes:ti.types.ndarray(dtype=ti.int64, ndim=1),
  opacity:ti.types.ndarray(ndim=1), 
  binomial_table:ti.types.ndarray(dtype=ti.f32, ndim=2), 
  n : ti.types.ndarray(dtype=ti.i64, ndim=1),

  out_opacity:ti.types.ndarray(ndim=1), 
  out_scale:ti.types.ndarray(ti.f32, ndim=1)):
  
  for i in range(indexes):
    idx = indexes[i]

    n = n[idx]
    denom_sum = 0.0

    new_opacity = 1.0 - ti.pow(1. - opacity, 1. / n)

    for i in range(n):
      for k in range(i):
        bin_coeff = binomial_table[i, k]
        
        term = (ti.pow(-1, k) / ti.sqrt(k + 1)) * ti.pow(new_opacity, k + 1)
        denom_sum += (bin_coeff * term)

      out_scale[i] = (opacity[idx] / denom_sum)
      out_opacity[i] = new_opacity


def compute_split(gaussians, probs, n):



# __global__ void compute_relocation(
#     int P, 
#     float* opacity_old, 
#     float* scale_old, 
#     int* N, 
#     float* binoms, 
#     int n_max, 
#     float* opacity_new, 
#     float* scale_new) 
# {
#     int idx = threadIdx.x + blockIdx.x * blockDim.x;
#     if (idx >= P) return;
    
#     int N_idx = N[idx];
#     float denom_sum = 0.0f;

#     // compute new opacity
#     opacity_new[idx] = 1.0f - powf(1.0f - opacity_old[idx], 1.0f / N_idx);
    
#     // compute new scale
#     for (int i = 1; i <= N_idx; ++i) {
#         for (int k = 0; k <= (i-1); ++k) {
#             float bin_coeff = binoms[(i-1) * n_max + k];
#             float term = (pow(-1, k) / sqrt(k + 1)) * pow(opacity_new[idx], k + 1);
#             denom_sum += (bin_coeff * term);
#         }
#     }
#     float coeff = (opacity_old[idx] / denom_sum);
#     for (int i = 0; i < 3; ++i)
#         scale_new[idx * 3 + i] = coeff * scale_old[idx * 3 + i];
# }