// fractional_adam.slang
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
    float bias_factor = 1;
    if (bias_correction == 1) {
        bias_factor = sqrt(1 - pow(beta2, total_weight[index])) / (1 - pow(beta1, total_weight[index]));
    }

    for(int d=0; d<D; d++) {

        float g = grad[index, d];

        int[2] i = {index, d};

        float m = lerp(pow(beta1, weight), m_arr[index, d], g);
        float v = lerp(pow(beta2, weight), v_arr[index, d], g * g);
        
        float sqrt_v = sqrt(v);
        if (sqrt_v < eps) {
            sqrt_v = eps;
        }

        lr_step[d] = m / sqrt_v * bias_factor * lr;
        
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

    int bias_correction,
    )
{   
    float bias_factor = 1;
    if (bias_correction == 1) {
        bias_factor = sqrt(1 - pow(beta2, total_weight[index])) / (1 - pow(beta1, total_weight[index]));
    }

    float norm = 0;
    for(int d=0; d<D; d++) {
        norm = norm + grad[index, d] * grad[index, d];
    }

    float[D] m;
    for(int d=0; d<D; d++) {
        m[d] = lerp(pow(beta1, weight), m_arr[index, d], grad[index, d]);
    }

    float v = lerp(pow(beta1, weight), v_arr[index], norm);

    float sqrt_v = sqrt(v);
    if (sqrt_v < eps) {
        sqrt_v = eps;
    }

    for(int d=0; d<D; d++) {
        lr_step[d] = m[d] / sqrt_v * bias_factor * lr;

        int[2] i2 = {index, d};
        m_arr.set(i2, m[d]);
    }

    int[1] i1 = {index};
    v_arr.set(i1, v);
}
