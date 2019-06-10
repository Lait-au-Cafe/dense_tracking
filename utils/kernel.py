# -*- coding: utf-8 -*-
import cupy as cp

blend_images = cp.RawKernel(r'''
extern "C" __global__
void blend_images(
    const float* input0, 
    const float* input1, 
    float3* output, 
    int width, 
    int height
) {
    const int tx = blockIdx.x*blockDim.x + threadIdx.x;
    const int ty = blockIdx.y*blockDim.y + threadIdx.y;

    if(tx >= width || ty >= height) { return; }

    const int index = tx + ty * width;

    output[index] = make_float3(input0[index], input1[index], 0);
}
''', 'blend_images')

warp_and_blend_images = cp.RawKernel(r'''
extern "C" __global__
void warp_and_blend_images(
    const float* input0, 
    const float* input1, 
    const double* warp_mat, 
    float3* output, 
    int width, 
    int height
) {
    const int tx = blockIdx.x*blockDim.x + threadIdx.x;
    const int ty = blockIdx.y*blockDim.y + threadIdx.y;

    if(tx >= width || ty >= height) { return; }

    const float wz = warp_mat[6] * tx + warp_mat[7] * ty + warp_mat[8];
    const int wx = (warp_mat[0] * tx + warp_mat[1] * ty + warp_mat[2]) / wz;
    const int wy = (warp_mat[3] * tx + warp_mat[4] * ty + warp_mat[5]) / wz;

    float w_val = 0;
    const int w_index = wx + wy * width;
    if(wx >= 0 && wx < width && wy >= 0 && wy < height) {
        w_val = input0[w_index];
    }

    const int index = tx + ty * width;

    output[index] = make_float3(w_val, input1[index], 0);
}
''', 'warp_and_blend_images')

step_minimization = cp.RawKernel(r'''
inline __device__ double atomicAdd(
    double* addr, 
    double val
) {
    unsigned long long int* addr_as_ull = (unsigned long long int*)addr;
    unsigned long long int old = *addr_as_ull;
    unsigned long long int assumed;
    do {
        assumed = old;
        old = atomicCAS(
            addr_as_ull, 
            assumed, 
            __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old); // Note: uses int comparison to avoid hang-in-case of NaN

    return __longlong_as_double(old);
}

extern "C" __global__
void step_minimization(
    const float* LivImg, 
    const float* RefImg, 
    int width, 
    int height, 
    const float* WarpMat, 
    float* GenMat, 
    int dim_param, 
    double* MatA, 
    double* Matb, 
    double* Error
) {
    const int tx = blockIdx.x*blockDim.x + threadIdx.x;
    const int ty = blockIdx.y*blockDim.y + threadIdx.y;

    if(tx >= width || ty >= height) { return; }

    const float wz =  WarpMat[6] * tx + WarpMat[7] * ty + WarpMat[8];
    const int   wx = (WarpMat[0] * tx + WarpMat[1] * ty + WarpMat[2]) / wz;
    const int   wy = (WarpMat[3] * tx + WarpMat[4] * ty + WarpMat[5]) / wz;

    if(wx < 0 || wx >= width || wy < 0 || wy >= height) { return; }

    float diff = LivImg[wx + wy * width] - RefImg[tx + ty * width];
    double err = 0.5  * diff * diff;
    atomicAdd(Error, err);

    const float depth = 1.0e-20;
    float Ix = (RefImg[min(tx + 1, width - 1) + ty * width] 
                    - RefImg[max(tx - 1, 0) + ty * width]) * 0.5;
    float Iy = (RefImg[tx + min(ty + 1, height - 1) * width] 
                    - RefImg[tx + max(ty - 1, 0) * width]) * 0.5;
    float Iz = (tx * Ix + ty * Iy) / depth;

    int i, j;
    float* gen_mat;
    double elm_A, elm_b;
    double param[9]; // dim_param <= 9
    for(i = 0; i < dim_param; i++) {
        gen_mat = GenMat + 9 * i;
        param[i] =  - Ix * (gen_mat[0] * tx + gen_mat[1] * ty + gen_mat[2]) 
                    - Iy * (gen_mat[3] * tx + gen_mat[4] * ty + gen_mat[5]) 
                    + Iz * (gen_mat[6] * tx + gen_mat[7] * ty + gen_mat[8]);
        elm_b = -diff * param[i];
        atomicAdd(&Matb[i], elm_b);
    }

    for(j = 0; j < dim_param; j++) {
        for(i = 0; i < dim_param; i++) {
            elm_A = param[i] * param[j];
            atomicAdd(&MatA[i + j * dim_param], elm_A);
        }
    }
}
''', 'step_minimization')
