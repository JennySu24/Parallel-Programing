#pragma once

#include <cuda_runtime.h>

// CUDA 核函数的 Host 封装
void launch_generate_single_segment(char **values, char *output, int count, int blockSize, int gridSize);
void launch_generate_multi_segment(char *prefix, char **suffixes, char *output, int count, int blockSize, int gridSize);


