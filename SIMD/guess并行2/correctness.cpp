#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
using namespace std;
using namespace chrono;

// 编译指令如下：
// g++ correctness.cpp train.cpp guessing.cpp md5.cpp -o main


// 通过这个函数，你可以验证你实现的SIMD哈希函数的正确性
int main()
{
    uint32x2_t state[4];
    MD5Hash("bvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfk","ajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdf", state);
    for (int i = 0; i < 4; ++i)
        cout << hex << setw(8) << setfill('0') << vget_lane_u32(state[i], 0);
    cout << endl;
    for (int i = 0; i < 4; ++i)
        cout << hex << setw(8) << setfill('0') << vget_lane_u32(state[i], 1);
    cout << endl;
}