#include <cuda_runtime.h>
#include "guessing_kernel.h"
#include "PCFG.h"
#define MAX_STR_LEN 64


__global__ void GenerateGuessesKernel(
    const char *prefixes,
    const char *values,
    const int *prefix_lens,
    const int *value_lens,
    const int *cumulative_counts,
    int n_pts,
    int total_guesses,
    int max_guess_len,
    char *output)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_guesses) return;

    // 找到它属于哪个 PT
    int pt_idx = 0;
    while (pt_idx < n_pts && cumulative_counts[pt_idx + 1] <= idx) pt_idx++;

    int value_idx = idx - cumulative_counts[pt_idx];

    const char *prefix = prefixes + pt_idx * max_guess_len;
    const char *value  = values + cumulative_counts[pt_idx] * max_guess_len + value_idx * max_guess_len;

    int prefix_len = prefix_lens[pt_idx];
    int value_len  = value_lens[pt_idx];

    char *guess = output + idx * max_guess_len;

    memcpy(guess, prefix, prefix_len);
    memcpy(guess + prefix_len, value, value_len);
    guess[prefix_len + value_len] = '\0';
}

void PriorityQueue::Generate_GPU(const vector<PT>& pts, vector<string>& guesses_out) {
    int n = pts.size();
    if (n == 0) return;

    // 1. 预处理：计算所有必要的信息
    vector<string> all_prefixes, all_values;
    vector<int> prefix_lens, value_lens, value_counts, cumulative_counts(n + 1, 0);
    int max_guess_len = 0;

    for (int i = 0; i < n; ++i) {
        const auto& pt = pts[i];
        string prefix;
        if (pt.content.size() > 1) {
            int seg_idx = 0;
            for (int idx : pt.curr_indices) {
                if (pt.content[seg_idx].type == 1)
                    prefix += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
                else if (pt.content[seg_idx].type == 2)
                    prefix += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
                else
                    prefix += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
                seg_idx++;
                if (seg_idx == pt.content.size() - 1) break;
            }
        }
        segment* last_seg = nullptr;
        if (pt.content.back().type == 1) 
            last_seg = &m.letters[m.FindLetter(pt.content.back())];
        else if (pt.content.back().type == 2) 
            last_seg = &m.digits[m.FindDigit(pt.content.back())];
        else 
            last_seg = &m.symbols[m.FindSymbol(pt.content.back())];
        int count = (pt.content.size() == 1) ? pt.max_indices[0] : pt.max_indices.back();
        int value_len = last_seg->ordered_values.empty() ? 0 : last_seg->ordered_values[0].size();
        all_prefixes.push_back(prefix);
        prefix_lens.push_back(prefix.size());
        value_lens.push_back(value_len);
        value_counts.push_back(count);
        cumulative_counts[i + 1] = cumulative_counts[i] + count;
        max_guess_len = max(max_guess_len, (int)prefix.size() + value_len);
        for (int j = 0; j < count; ++j) {
            all_values.push_back(last_seg->ordered_values[j]);
        }
    }
    int total_guesses = cumulative_counts[n];
    max_guess_len = max(max_guess_len, 1);

    // 2. 分配主机内存
    char *h_prefixes = new char[n * max_guess_len];
    char *h_values = new char[total_guesses * max_guess_len];
    char *h_output = new char[total_guesses * max_guess_len];

    // 数据打包
    for (int i = 0; i < n; ++i) {
        memset(h_prefixes + i * max_guess_len, 0, max_guess_len);
        memcpy(h_prefixes + i * max_guess_len, all_prefixes[i].c_str(), all_prefixes[i].size());
    }
    for (int i = 0; i < total_guesses; ++i) {
        memset(h_values + i * max_guess_len, 0, max_guess_len);
        memcpy(h_values + i * max_guess_len, all_values[i].c_str(), all_values[i].size());
    }

    // 3. 分配设备内存
    char *d_prefixes, *d_values, *d_output;
    int *d_prefix_lens, *d_value_lens, *d_cumulative_counts;
    
    cudaMalloc(&d_prefixes, n * max_guess_len);
    cudaMalloc(&d_values, total_guesses * max_guess_len);
    cudaMalloc(&d_output, total_guesses * max_guess_len);
    cudaMalloc(&d_prefix_lens, n * sizeof(int));
    cudaMalloc(&d_value_lens, n * sizeof(int));
    cudaMalloc(&d_cumulative_counts, (n + 1) * sizeof(int));

    // 4. 数据传输
    cudaMemcpy(d_prefixes, h_prefixes, n * max_guess_len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values, total_guesses * max_guess_len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_prefix_lens, prefix_lens.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_value_lens, value_lens.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cumulative_counts, cumulative_counts.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice);

    // 5. 启动kernel
    int block_size = 256;
    int grid_size = (total_guesses + block_size - 1) / block_size;
    
    GenerateGuessesKernel<<<grid_size, block_size>>>(
        d_prefixes, d_values, d_prefix_lens, d_value_lens, 
        d_cumulative_counts, n, total_guesses, max_guess_len, d_output);

    // 6. 回传结果
    cudaMemcpy(h_output, d_output, total_guesses * max_guess_len, cudaMemcpyDeviceToHost);

    // 7. 构建结果
    guesses_out.clear();
    guesses_out.reserve(total_guesses);
    for (int i = 0; i < total_guesses; ++i) {
        int pt_idx = 0;
        while (pt_idx < n && cumulative_counts[pt_idx + 1] <= i) pt_idx++;
        int guess_len = prefix_lens[pt_idx] + value_lens[pt_idx];
        guesses_out.emplace_back(h_output + i * max_guess_len, guess_len);
    }

    // 8. 释放资源
    delete[] h_prefixes;
    delete[] h_values;
    delete[] h_output;
    cudaFree(d_prefixes);
    cudaFree(d_values);
    cudaFree(d_output);
    cudaFree(d_prefix_lens);
    cudaFree(d_value_lens);
    cudaFree(d_cumulative_counts);
}


