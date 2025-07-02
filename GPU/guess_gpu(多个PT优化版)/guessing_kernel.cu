#include <cuda_runtime.h>
#include "guessing_kernel.h"
#include "PCFG.h"
#define MIN_GPU_BATCH_SIZE 10000


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


// void PriorityQueue::Generate_GPU(const vector<PT>& pts, vector<string>& guesses_out) {
//     int n = pts.size();
//     if (n == 0) return;

//     vector<string> all_prefixes, all_values;
//     vector<int> prefix_lens, value_lens, value_counts, cumulative_counts(n + 1, 0);
//     int max_guess_len = 0;

//     for (int i = 0; i < n; ++i) {
//         const auto& pt = pts[i];
//         string prefix;
//         if (pt.content.size() > 1) {
//             int seg_idx = 0;
//             for (int idx : pt.curr_indices) {
//                 if (pt.content[seg_idx].type == 1)
//                     prefix += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
//                 else if (pt.content[seg_idx].type == 2)
//                     prefix += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
//                 else
//                     prefix += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
//                 seg_idx++;
//                 if (seg_idx == pt.content.size() - 1) break;
//             }
//         }
//         segment* last_seg = nullptr;
//         if (pt.content.back().type == 1)
//             last_seg = &m.letters[m.FindLetter(pt.content.back())];
//         else if (pt.content.back().type == 2)
//             last_seg = &m.digits[m.FindDigit(pt.content.back())];
//         else
//             last_seg = &m.symbols[m.FindSymbol(pt.content.back())];
//         int count = (pt.content.size() == 1) ? pt.max_indices[0] : pt.max_indices.back();
//         int value_len = last_seg->ordered_values.empty() ? 0 : last_seg->ordered_values[0].size();
//         all_prefixes.push_back(prefix);
//         prefix_lens.push_back(prefix.size());
//         value_lens.push_back(value_len);
//         value_counts.push_back(count);
//         cumulative_counts[i + 1] = cumulative_counts[i] + count;
//         max_guess_len = max(max_guess_len, (int)prefix.size() + value_len);
//         for (int j = 0; j < count; ++j) {
//             all_values.push_back(last_seg->ordered_values[j]);
//         }
//     }

//     int total_guesses = cumulative_counts[n];
//     max_guess_len = ((max_guess_len + 3) / 4) * 4; // 保证4字节对齐

//     // 分配主机内存
//     char* h_prefixes = new char[n * max_guess_len];
//     char* h_values = new char[total_guesses * max_guess_len];
//     char* h_output = new char[total_guesses * max_guess_len];

//     // ----------------------------
//     // 向量化拷贝：prefixes
//     // ----------------------------
//     for (int i = 0; i < n; ++i) {
//         memset(h_prefixes + i * max_guess_len, 0, max_guess_len);
//         const char* src = all_prefixes[i].c_str();
//         char* dst = h_prefixes + i * max_guess_len;
//         int len = all_prefixes[i].size();
//         int aligned = len / 4;
//         for (int j = 0; j < aligned; ++j) {
//             reinterpret_cast<int*>(dst)[j] = reinterpret_cast<const int*>(src)[j];
//         }
//         for (int j = aligned * 4; j < len; ++j) {
//             dst[j] = src[j];
//         }
//     }

//     // ----------------------------
//     // 向量化拷贝：values
//     // ----------------------------
//     for (int i = 0; i < total_guesses; ++i) {
//         memset(h_values + i * max_guess_len, 0, max_guess_len);
//         const char* src = all_values[i].c_str();
//         char* dst = h_values + i * max_guess_len;
//         int len = all_values[i].size();
//         int aligned = len / 4;
//         for (int j = 0; j < aligned; ++j) {
//             reinterpret_cast<int*>(dst)[j] = reinterpret_cast<const int*>(src)[j];
//         }
//         for (int j = aligned * 4; j < len; ++j) {
//             dst[j] = src[j];
//         }
//     }

//     // 分配设备内存
//     char *d_prefixes, *d_values, *d_output;
//     int *d_prefix_lens, *d_value_lens, *d_cumulative_counts;

//     cudaMalloc(&d_prefixes, n * max_guess_len);
//     cudaMalloc(&d_values, total_guesses * max_guess_len);
//     cudaMalloc(&d_output, total_guesses * max_guess_len);
//     cudaMalloc(&d_prefix_lens, n * sizeof(int));
//     cudaMalloc(&d_value_lens, n * sizeof(int));
//     cudaMalloc(&d_cumulative_counts, (n + 1) * sizeof(int));

//     cudaMemcpy(d_prefixes, h_prefixes, n * max_guess_len, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_values, h_values, total_guesses * max_guess_len, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_prefix_lens, prefix_lens.data(), n * sizeof(int), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_value_lens, value_lens.data(), n * sizeof(int), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_cumulative_counts, cumulative_counts.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice);

//     // 启动 Kernel
//     int block_size = 256;
//     int grid_size = (total_guesses + block_size - 1) / block_size;

//     GenerateGuessesKernel<<<grid_size, block_size>>>(
//         d_prefixes, d_values, d_prefix_lens, d_value_lens,
//         d_cumulative_counts, n, total_guesses, max_guess_len, d_output
//     );

//     cudaMemcpy(h_output, d_output, total_guesses * max_guess_len, cudaMemcpyDeviceToHost);

//     // 回收结果
//     guesses_out.clear();
//     guesses_out.reserve(total_guesses);
//     for (int i = 0; i < total_guesses; ++i) {
//         int pt_idx = 0;
//         while (pt_idx < n && cumulative_counts[pt_idx + 1] <= i) pt_idx++;
//         int guess_len = prefix_lens[pt_idx] + value_lens[pt_idx];
//         guesses_out.emplace_back(h_output + i * max_guess_len, guess_len);
//     }

//     delete[] h_prefixes;
//     delete[] h_values;
//     delete[] h_output;
//     cudaFree(d_prefixes);
//     cudaFree(d_values);
//     cudaFree(d_output);
//     cudaFree(d_prefix_lens);
//     cudaFree(d_value_lens);
//     cudaFree(d_cumulative_counts);
// }
// 二分查找
__device__ __forceinline__ int binary_search_pt(const int* cumulative_counts, int n, int idx) {
    int left = 0, right = n;
    while (left < right) {
        int mid = (left + right) >> 1;
        if (cumulative_counts[mid + 1] <= idx)
            left = mid + 1;
        else
            right = mid;
    }
    return left;
}
__global__ void GenerateGuessesOptimized(
    const char* __restrict__ prefixes,
    const char* __restrict__ values, 
    const int* __restrict__ prefix_lens,
    const int* __restrict__ value_lens,
    const int* __restrict__ cumulative_counts,
    int n_pts,
    int total_guesses,
    int max_guess_len,
    char* __restrict__ output)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_guesses) return;

    // 二分查找PT索引
    int pt_idx = binary_search_pt(cumulative_counts, n_pts, idx);
    int local_idx = idx - cumulative_counts[pt_idx];
    int prefix_len = prefix_lens[pt_idx];
    int value_len = value_lens[pt_idx];

    const char* prefix_ptr = prefixes + pt_idx * max_guess_len;
    const char* value_ptr = values + (cumulative_counts[pt_idx] + local_idx) * value_len;
    char* out_ptr = output + idx * max_guess_len;

    // 合并内存拷贝
    int copy_len = prefix_len + value_len;

    // 拷贝前缀
    for (int i = 0; i < prefix_len; ++i) out_ptr[i] = prefix_ptr[i];
    // 拷贝值
    for (int i = 0; i < value_len; ++i) out_ptr[prefix_len + i] = value_ptr[i];
    // 清零剩余部分
    for (int i = copy_len; i < max_guess_len; ++i) out_ptr[i] = 0;
}
// 内存对齐优化
inline int align_to(int n, int alignment) { 
    return ((n + alignment - 1) / alignment) * alignment; 
}
GPUMemoryPool& get_memory_pool() {
    static GPUMemoryPool pool;
    return pool;
}
// 增加全局pinned memory池，避免频繁分配释放
struct GPUPinnedBuffer {
    char* prefixes = nullptr;
    char* values = nullptr;
    char* output = nullptr;
    size_t prefixes_size = 0;
    size_t values_size = 0;
    size_t output_size = 0;
    ~GPUPinnedBuffer() {
        if (prefixes) cudaFreeHost(prefixes);
        if (values) cudaFreeHost(values);
        if (output) cudaFreeHost(output);
    }
    void alloc(size_t pre_sz, size_t val_sz, size_t out_sz) {
        if (pre_sz > prefixes_size) {
            if (prefixes) cudaFreeHost(prefixes);
            cudaMallocHost(&prefixes, pre_sz);
            prefixes_size = pre_sz;
        }
        if (val_sz > values_size) {
            if (values) cudaFreeHost(values);
            cudaMallocHost(&values, val_sz);
            values_size = val_sz;
        }
        if (out_sz > output_size) {
            if (output) cudaFreeHost(output);
            cudaMallocHost(&output, out_sz);
            output_size = out_sz;
        }
    }
};
static GPUPinnedBuffer g_pinned_buf;
void PriorityQueue::Generate_GPU(const vector<PT>& pts, vector<string>& guesses_out) {
    int n = pts.size();
    if (n == 0) return;

    // 1. 快速评估：如果总猜测数太少，直接用CPU
    int total_estimate = 0;
    for (const auto& pt : pts) {
        int count = (pt.content.size() == 1) ? pt.max_indices[0] : pt.max_indices.back();
        total_estimate += count;
        if (total_estimate > MIN_GPU_BATCH_SIZE) break;
    }
    if (total_estimate < MIN_GPU_BATCH_SIZE) {
        Generate(pts, guesses_out);
        return;
    }

    auto& pool = get_memory_pool();

    // 2. 预处理：计算所有必要的信息

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
    max_guess_len = align_to(max_guess_len, 4);


    // 3. 高效的内存分配和数据传输

    // 复用全局 pinned buffer
    g_pinned_buf.alloc(n * max_guess_len, total_guesses * max_guess_len, total_guesses * max_guess_len);
    char *h_prefixes_pinned = g_pinned_buf.prefixes;
    char *h_values_pinned = g_pinned_buf.values;
    char *h_output_pinned = g_pinned_buf.output;

    // 数据打包
    if (total_guesses > 100000) {
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            memset(h_prefixes_pinned + i * max_guess_len, 0, max_guess_len);
            memcpy(h_prefixes_pinned + i * max_guess_len, all_prefixes[i].c_str(), all_prefixes[i].size());
        }
        #pragma omp parallel for
        for (int i = 0; i < total_guesses; ++i) {
            memset(h_values_pinned + i * max_guess_len, 0, max_guess_len);
            memcpy(h_values_pinned + i * max_guess_len, all_values[i].c_str(), all_values[i].size());
        }
    } else {
        for (int i = 0; i < n; ++i) {
            memset(h_prefixes_pinned + i * max_guess_len, 0, max_guess_len);
            memcpy(h_prefixes_pinned + i * max_guess_len, all_prefixes[i].c_str(), all_prefixes[i].size());
        }
        for (int i = 0; i < total_guesses; ++i) {
            memset(h_values_pinned + i * max_guess_len, 0, max_guess_len);
            memcpy(h_values_pinned + i * max_guess_len, all_values[i].c_str(), all_values[i].size());
        }
    }

    char* d_prefixes = (char*)pool.allocate(n * max_guess_len);
    char* d_values = (char*)pool.allocate(total_guesses * max_guess_len);
    char* d_output = (char*)pool.allocate(total_guesses * max_guess_len);
    int* d_prefix_lens = (int*)pool.allocate(n * sizeof(int));
    int* d_value_lens = (int*)pool.allocate(n * sizeof(int));
    int* d_cumulative_counts = (int*)pool.allocate((n + 1) * sizeof(int));

    cudaStream_t stream = pool.get_stream();
    cudaMemcpyAsync(d_prefixes, h_prefixes_pinned, n * max_guess_len, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_values, h_values_pinned, total_guesses * max_guess_len, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_prefix_lens, prefix_lens.data(), n * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_value_lens, value_lens.data(), n * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_cumulative_counts, cumulative_counts.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice, stream);


    // 4. 优化的kernel启动

    int device_id;
    cudaGetDevice(&device_id);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    int max_threads_per_block = prop.maxThreadsPerBlock;
    int block_size = min(1024, max_threads_per_block);
    int grid_size = (total_guesses + block_size - 1) / block_size;

    GenerateGuessesKernel<<<grid_size, block_size, 0, stream>>>(
        d_prefixes, d_values, d_prefix_lens, d_value_lens, 
        d_cumulative_counts, n, total_guesses, max_guess_len, d_output);

    // 5. 高效的结果回传
    cudaMemcpyAsync(h_output_pinned, d_output, total_guesses * max_guess_len, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    guesses_out.clear();
    guesses_out.reserve(total_guesses);

    // 并行构建结果字符串
    #pragma omp parallel for
    for (int i = 0; i < total_guesses; ++i) {
        int pt_idx = 0;
        // 二分查找pt_idx
        int l = 0, r = n;
        while (l < r) {
            int mid = (l + r) >> 1;
            if (cumulative_counts[mid + 1] <= i)
                l = mid + 1;
            else
                r = mid;
        }
        pt_idx = l;
        int actual_len = prefix_lens[pt_idx] + value_lens[pt_idx];
        guesses_out.emplace_back(h_output_pinned + i * max_guess_len, actual_len);
    }


    // 6. 清理资源
    pool.deallocate(d_prefixes);
    pool.deallocate(d_values);
    pool.deallocate(d_output);
    pool.deallocate(d_prefix_lens);
    pool.deallocate(d_value_lens);
    pool.deallocate(d_cumulative_counts);

}




