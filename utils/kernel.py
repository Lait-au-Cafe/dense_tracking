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
