#include <cuda_runtime.h>
#include "guessing_kernel.h"
#define MAX_STR_LEN 64



__global__ void generate_single_segment_kernel(char **values, char *output, int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    char *src = values[idx];
    char *dst = output + idx * MAX_STR_LEN;

    int i = 0;
    for (; i < MAX_STR_LEN - 1 && src[i] != '\0'; ++i)
        dst[i] = src[i];
    dst[i] = '\0';
}

__global__ void generate_multi_segment_kernel(char *prefix, char **suffixes, char *output, int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    char *suffix = suffixes[idx];
    char *dst = output + idx * MAX_STR_LEN;

    int i = 0;
    for (; i < MAX_STR_LEN - 1 && prefix[i] != '\0'; ++i)
        dst[i] = prefix[i];
    for (int j = 0; i < MAX_STR_LEN - 1 && suffix[j] != '\0' && i < MAX_STR_LEN - 1; ++j, ++i)
        dst[i] = suffix[j];
    dst[i] = '\0';
}

void launch_generate_single_segment(char **values, char *output, int count, int blockSize, int gridSize) {
    generate_single_segment_kernel<<<gridSize, blockSize>>>(values, output, count);
    cudaDeviceSynchronize();
}

void launch_generate_multi_segment(char *prefix, char **suffixes, char *output, int count, int blockSize, int gridSize) {
    generate_multi_segment_kernel<<<gridSize, blockSize>>>(prefix, suffixes, output, count);
    cudaDeviceSynchronize();
}




