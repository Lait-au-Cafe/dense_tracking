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
