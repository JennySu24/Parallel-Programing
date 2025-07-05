#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <emmintrin.h>  // SSE2
#include <smmintrin.h>  // SSE4.1 for _mm_extract_epi32
#include <tmmintrin.h>

using namespace std;
using namespace chrono;

// 编译指令如下：
// g++ -mssse3 correctness.cpp train.cpp guessing.cpp md5.cpp -o main
// g++ correctness.cpp train.cpp guessing.cpp md5.cpp -o main -lpthread -std=c++11
// g++ correctness.cpp train.cpp guessing.cpp md5.cpp -o main -fopenmp -std=c++11

// 通过这个函数，你可以验证你实现的SIMD哈希函数的正确性
int main()
{
    __m128i state[4];
    MD5Hash("nankaicskdlwurpf123jdh*4@", "juqodh93jwdosjfhalojf", "huiydckajeicghheslfeis9ve", "sjlhlsjhzjhgihrih927yjdjfgzfhgir", state);

    for (int i = 0; i < 4; ++i)
        cout << hex << setw(8) << setfill('0') << _mm_extract_epi32(state[i], 0);
    cout << endl;

    for (int i = 0; i < 4; ++i)
        cout << hex << setw(8) << setfill('0') << _mm_extract_epi32(state[i], 1);
    cout << endl;

    for (int i = 0; i < 4; ++i)
        cout << hex << setw(8) << setfill('0') << _mm_extract_epi32(state[i], 2);
    cout << endl;

    for (int i = 0; i < 4; ++i)
        cout << hex << setw(8) << setfill('0') << _mm_extract_epi32(state[i], 3);
    cout << endl;

    return 0;
}