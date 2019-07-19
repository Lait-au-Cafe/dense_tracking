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

step_minimization_integer = cp.RawKernel(r'''
inline __device__ int64_t __ull_as_ll(uint64_t val) {
    return *(int64_t*)&val;
}

inline __device__ uint64_t __ll_as_ull(int64_t val) {
    return *(uint64_t*)&val;
}

inline __device__ int64_t atomicAdd(
    int64_t* addr, 
    int64_t val
) {
    uint64_t* addr_as_ull = (uint64_t*)addr;
    uint64_t old = *addr_as_ull;
    uint64_t assumed;
    do {
        assumed = old;
        old = atomicCAS(
            addr_as_ull, 
            assumed, 
            __ll_as_ull(val + __ull_as_ll(assumed)));
    } while (assumed != old); // Note: uses int comparison to avoid hang-in-case of NaN

    return __ull_as_ll(old);
}

extern "C" __global__
void step_minimization_integer(
    const int32_t* LivImg, 
    const int32_t* RefImg, 
    int32_t width, 
    int32_t height, 
    const float* WarpMat, 
    float* GenMat, 
    int32_t dim_param, 
    int64_t* MatA, 
    int64_t* Matb, 
    int64_t* Error
) {
    const int32_t tx = blockIdx.x*blockDim.x + threadIdx.x;
    const int32_t ty = blockIdx.y*blockDim.y + threadIdx.y;

    if(tx >= width || ty >= height) { return; }

    const float   wz =  WarpMat[6] * tx + WarpMat[7] * ty + WarpMat[8];
    const int32_t wx = (WarpMat[0] * tx + WarpMat[1] * ty + WarpMat[2]) / wz;
    const int32_t wy = (WarpMat[3] * tx + WarpMat[4] * ty + WarpMat[5]) / wz;

    if(wx < 0 || wx >= width || wy < 0 || wy >= height) { return; }

    int32_t diff = LivImg[wx + wy * width] - RefImg[tx + ty * width];
    int64_t err = diff * diff;
    atomicAdd(Error, err);

    const int64_t scale = 1.0;
    int32_t Ix = RefImg[min(tx + 1, width - 1) + ty * width]-RefImg[max(tx - 1, 0) + ty * width];
    int32_t Iy = RefImg[tx + min(ty + 1, height -1) * width]-RefImg[tx + max(ty - 1, 0) * width];
    int32_t Iz = (tx * Ix + ty * Iy) * scale;

    int32_t i, j;
    float* gen_mat;
    int64_t elm_A, elm_b;
    int64_t param[9]; // dim_param <= 9
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
''', 'step_minimization_integer')
