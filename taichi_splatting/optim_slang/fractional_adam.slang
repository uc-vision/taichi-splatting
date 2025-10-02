// fractional_adam.slang
import "slangpy";
import "vector.slang";

[mutating]
void scalar_kernel<let D : int, let bias_correction : bool>(
    inout float[D] lr_step,
    int index,
    float weight,

    RWTensor<float, 2> m_arr,
    RWTensor<float, 2> v_arr,
    RWTensor<float, 1> total_weight,
    RWTensor<float, 2> grad,

    float lr,
    float beta1,
    float beta2,
    float eps,

    )
{   
    float bias_factor = 1;
    if (bias_correction) {
        float tw = total_weight[index];
        bias_factor = sqrt(1 - pow(beta2, tw)) / (1 - pow(beta1, tw));
    }

    let g = load_vector<float, D>(grad, index);
    let m = load_vector<float, D>(m_arr, index);
    let v = load_vector<float, D>(v_arr, index);

    let g2 = g * g;

    float beta1_pow = pow(beta1, weight);
    float beta2_pow = pow(beta2, weight);

    let m_new = m.lerp(g, beta1_pow);
    let v_new = v.lerp(g * g, beta2_pow);

    store_vector(m_arr, m_new, index);
    store_vector(v_arr, v_new, index);

    float scale = bias_factor * lr;
    lr_step = (m_new / (v_new.sqrt() + eps) * scale).data;

}



[mutating]
void vector_kernel<let D : int, let bias_correction : bool>(
    inout float[D] lr_step,
    int index,
    float weight,

    RWTensor<float, 2> m_arr,
    RWTensor<float, 1> v_arr,
    RWTensor<float, 1> total_weight,
    RWTensor<float, 2> grad,

    float lr,
    float beta1,
    float beta2,
    float eps,

    )
{   
    float bias_factor = 1;
    if (bias_correction) {
        bias_factor = sqrt(1 - pow(beta2, total_weight[index])) / (1 - pow(beta1, total_weight[index]));
    }

    let g = load_vector<float, D>(grad, index);
    let m = load_vector<float, D>(m_arr, index);

    float v = lerp(v_arr[index], g.norm(), pow(beta2, weight));

    let m_new = m.lerp(g, pow(beta1, weight));

    float scale = (lr * bias_factor) / max(sqrt(v), eps);

    [ForceUnroll]
    for (int d = 0; d < D; d++) {
        lr_step[d] = m_new.data[d] * scale;
    }

    store_vector(m_arr, m_new, index);
    v_arr.set({index}, v);

}
