// fractional_laprop.slang
import "slangpy";

float lerp(float t, float a, float b) {
    return a * t + b * (1 - t);
}

[mutating]
void scalar_kernel<let D : int>(
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

    int bias_correction,
    )
{   
    float bias1 = 1;
    float bias2 = 1;
    if (bias_correction == 1) {
        bias1 = 1 - pow(beta1, total_weight[index]);
        bias2 = 1 - pow(beta2, total_weight[index]);
    }

    for(int d=0; d<D; d++) {

        float g = grad[index, d];

        int[2] i = {index, d};

        float v = lerp(pow(beta2, weight), v_arr[index, d], g * g);
        float sqrt_v_b2 = sqrt(v / bias2);
        if (sqrt_v_b2 < eps) {
            sqrt_v_b2 = eps;
        }
        float m = lerp(pow(beta1, weight), m_arr[index, d], g / sqrt_v_b2);

        lr_step[d] = m * lr / bias1;
        
        m_arr.set(i, m);
        v_arr.set(i, v);
    }
}

[mutating]
void vector_kernel<let D : int>(
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

    int bias_correction
)
{
    float bias1 = 1.0;
    float bias2 = 1.0;

    if (bias_correction == 1) {
        bias1 = 1.0 - pow(beta1, total_weight[index]);
        bias2 = 1.0 - pow(beta2, total_weight[index]);
    }

    float norm_g_sq = 0;
    for (int d = 0; d < D; d++) {
        norm_g_sq += grad[index, d] * grad[index, d];
    }

    float v = lerp(pow(beta2, weight), v_arr[index], norm_g_sq);
    
    float v_div_b2 = v / bias2;
    float sqrt_v_div_b2 = sqrt(v_div_b2);

    if (sqrt_v_div_b2 < eps) {
        sqrt_v_div_b2 = eps;
    }

    float[D] m;
    for (int d = 0; d < D; d++) {
        float g_d = grad[index, d];
        m[d] = lerp(pow(beta1, weight), m_arr[index, d], g_d / sqrt_v_div_b2);
    }

    for (int d = 0; d < D; d++) {
        lr_step[d] = m[d] * lr / bias1;
        int[2] i2 = {index, d};
        m_arr.set(i2, m[d]);
    }

    int[1] i1 = {index};
    v_arr.set(i1, v);
}