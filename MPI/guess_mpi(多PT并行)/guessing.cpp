#include "PCFG.h"
#include <mpi.h>
#include <omp.h>
#include <sstream>
using namespace std;

void PriorityQueue::CalProb(PT &pt)
{
    // 计算PriorityQueue里面一个PT的流程如下：
    // 1. 首先需要计算一个PT本身的概率。例如，L6S1的概率为0.15
    // 2. 需要注意的是，Queue里面的PT不是“纯粹的”PT，而是除了最后一个segment以外，全部被value实例化的PT
    // 3. 所以，对于L6S1而言，其在Queue里面的实际PT可能是123456S1，其中“123456”为L6的一个具体value。
    // 4. 这个时候就需要计算123456在L6中出现的概率了。假设123456在所有L6 segment中的概率为0.1，那么123456S1的概率就是0.1*0.15

    // 计算一个PT本身的概率。后续所有具体segment value的概率，直接累乘在这个初始概率值上
    pt.prob = pt.preterm_prob;

    // index: 标注当前segment在PT中的位置
    int index = 0;


    for (int idx : pt.curr_indices)
    {
        // pt.content[index].PrintSeg();
        if (pt.content[index].type == 1)
        {
            // 下面这行代码的意义：
            // pt.content[index]：目前需要计算概率的segment
            // m.FindLetter(seg): 找到一个letter segment在模型中的对应下标
            // m.letters[m.FindLetter(seg)]：一个letter segment在模型中对应的所有统计数据
            // m.letters[m.FindLetter(seg)].ordered_values：一个letter segment在模型中，所有value的总数目
            pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq;
            // cout << m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.letters[m.FindLetter(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 2)
        {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
            // cout << m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.digits[m.FindDigit(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 3)
        {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].total_freq << endl;
        }
        index += 1;
    }
    // cout << pt.prob << endl;
}

void PriorityQueue::init()
{
    // cout << m.ordered_pts.size() << endl;
    // 用所有可能的PT，按概率降序填满整个优先队列
    for (PT pt : m.ordered_pts)
    {
        for (segment seg : pt.content)
        {
            if (seg.type == 1)
            {
                // 下面这行代码的意义：
                // max_indices用来表示PT中各个segment的可能数目。例如，L6S1中，假设模型统计到了100个L6，那么L6对应的最大下标就是99
                // （但由于后面采用了"<"的比较关系，所以其实max_indices[0]=100）
                // m.FindLetter(seg): 找到一个letter segment在模型中的对应下标
                // m.letters[m.FindLetter(seg)]：一个letter segment在模型中对应的所有统计数据
                // m.letters[m.FindLetter(seg)].ordered_values：一个letter segment在模型中，所有value的总数目
                pt.max_indices.emplace_back(m.letters[m.FindLetter(seg)].ordered_values.size());
            }
            if (seg.type == 2)
            {
                pt.max_indices.emplace_back(m.digits[m.FindDigit(seg)].ordered_values.size());
            }
            if (seg.type == 3)
            {
                pt.max_indices.emplace_back(m.symbols[m.FindSymbol(seg)].ordered_values.size());
            }
        }
        pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;
        // pt.PrintPT();
        // cout << " " << m.preterm_freq[m.FindPT(pt)] << " " << m.total_preterm << " " << pt.preterm_prob << endl;

        // 计算当前pt的概率
        CalProb(pt);
        // 将PT放入优先队列
        priority.emplace_back(pt);
    }
    // cout << "priority size:" << priority.size() << endl;
}

void PriorityQueue::PopNext()
{

    // 对优先队列最前面的PT，首先利用这个PT生成一系列猜测
    Generate(priority.front());

    // 然后需要根据即将出队的PT，生成一系列新的PT
    vector<PT> new_pts = priority.front().NewPTs();
    for (PT pt : new_pts)
    {
        // 计算概率
        CalProb(pt);
        // 接下来的这个循环，作用是根据概率，将新的PT插入到优先队列中
        for (auto iter = priority.begin(); iter != priority.end(); iter++)
        {
            // 对于非队首和队尾的特殊情况
            if (iter != priority.end() - 1 && iter != priority.begin())
            {
                // 判定概率
                if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob)
                {
                    priority.emplace(iter + 1, pt);
                    break;
                }
            }
            if (iter == priority.end() - 1)
            {
                priority.emplace_back(pt);
                break;
            }
            if (iter == priority.begin() && iter->prob < pt.prob)
            {
                priority.emplace(iter, pt);
                break;
            }
        }
    }

    // 现在队首的PT善后工作已经结束，将其出队（删除）
    priority.erase(priority.begin());
}

// 这个函数你就算看不懂，对并行算法的实现影响也不大
// 当然如果你想做一个基于多优先队列的并行算法，可能得稍微看一看了
vector<PT> PT::NewPTs()
{
    // 存储生成的新PT
    vector<PT> res;

    // 假如这个PT只有一个segment
    // 那么这个segment的所有value在出队前就已经被遍历完毕，并作为猜测输出
    // 因此，所有这个PT可能对应的口令猜测已经遍历完成，无需生成新的PT
    if (content.size() == 1)
    {
        return res;
    }
    else
    {
        // 最初的pivot值。我们将更改位置下标大于等于这个pivot值的segment的值（最后一个segment除外），并且一次只更改一个segment
        // 上面这句话里是不是有没看懂的地方？接着往下看你应该会更明白
        int init_pivot = pivot;

        // 开始遍历所有位置值大于等于init_pivot值的segment
        // 注意i < curr_indices.size() - 1，也就是除去了最后一个segment（这个segment的赋值预留给并行环节）
        for (int i = pivot; i < curr_indices.size() - 1; i += 1)
        {
            // curr_indices: 标记各segment目前的value在模型里对应的下标
            curr_indices[i] += 1;

            // max_indices：标记各segment在模型中一共有多少个value
            if (curr_indices[i] < max_indices[i])
            {
                // 更新pivot值
                pivot = i;
                res.emplace_back(*this);
            }

            // 这个步骤对于你理解pivot的作用、新PT生成的过程而言，至关重要
            curr_indices[i] -= 1;
        }
        pivot = init_pivot;
        return res;
    }

    return res;
}


// 这个函数是PCFG并行化算法的主要载体
// 尽量看懂，然后进行并行实现
void PriorityQueue::Generate(PT pt)
{
    CalProb(pt);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (pt.content.size() == 1)
    {
        segment *a = nullptr;
        if (pt.content[0].type == 1)
            a = &m.letters[m.FindLetter(pt.content[0])];
        else if (pt.content[0].type == 2)
            a = &m.digits[m.FindDigit(pt.content[0])];
        else if (pt.content[0].type == 3)
            a = &m.symbols[m.FindSymbol(pt.content[0])];

        const vector<string>& ordered_values = a->ordered_values;
        int n = pt.max_indices[0];

        vector<int> counts(size, n / size);
        vector<int> displs(size, 0);
        int rem = n % size;
        for (int i = 0; i < rem; ++i) counts[i]++;
        for (int i = 1; i < size; ++i) displs[i] = displs[i - 1] + counts[i - 1];

        int local_n = counts[rank];
        int start = displs[rank];

        std::ostringstream oss;
        for (int i = 0; i < local_n; ++i)
        {
            oss << ordered_values[start + i] << '\n';
        }
        std::string payload = oss.str();
        int len = payload.size();

        if (rank == 0)
        {
            guesses.reserve(guesses.size() + n);
            // 收集本地生成
            std::istringstream iss(payload);
            std::string line;
            while (std::getline(iss, line))
                guesses.push_back(line);

            // 收集各进程数据
            for (int src = 1; src < size; ++src)
            {
                int recv_len;
                MPI_Recv(&recv_len, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                std::vector<char> buf(recv_len + 1, '\0');
                MPI_Recv(buf.data(), recv_len, MPI_CHAR, src, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                std::istringstream iss2(std::string(buf.data(), recv_len));
                while (std::getline(iss2, line))
                    guesses.push_back(line);
            }
            total_guesses += n;
        }
        else
        {
            MPI_Send(&len, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            MPI_Send(payload.data(), len, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
        }
    }
    else
    {
        // 前缀组合部分
        std::string prefix;
        int seg_idx = 0;
        for (int idx : pt.curr_indices)
        {
            if (pt.content[seg_idx].type == 1)
                prefix += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            else if (pt.content[seg_idx].type == 2)
                prefix += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            else if (pt.content[seg_idx].type == 3)
                prefix += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];

            seg_idx += 1;
            if (seg_idx == pt.content.size() - 1)
                break;
        }

        segment *a = nullptr;
        if (pt.content.back().type == 1)
            a = &m.letters[m.FindLetter(pt.content.back())];
        else if (pt.content.back().type == 2)
            a = &m.digits[m.FindDigit(pt.content.back())];
        else if (pt.content.back().type == 3)
            a = &m.symbols[m.FindSymbol(pt.content.back())];

        const vector<string>& last_values = a->ordered_values;
        int m_last = pt.max_indices.back();

        vector<int> counts(size, m_last / size);
        vector<int> displs(size, 0);
        int rem = m_last % size;
        for (int i = 0; i < rem; ++i) counts[i]++;
        for (int i = 1; i < size; ++i) displs[i] = displs[i - 1] + counts[i - 1];

        int local_m = counts[rank];
        int start = displs[rank];

        std::ostringstream oss;
        for (int i = 0; i < local_m; ++i)
        {
            oss << prefix + last_values[start + i] << '\n';
        }
        std::string payload = oss.str();
        int len = payload.size();

        if (rank == 0)
        {
            guesses.reserve(guesses.size() + m_last);

            std::istringstream iss(payload);
            std::string line;
            while (std::getline(iss, line))
                guesses.push_back(line);

            for (int src = 1; src < size; ++src)
            {
                int recv_len;
                MPI_Recv(&recv_len, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                std::vector<char> buf(recv_len + 1, '\0');
                MPI_Recv(buf.data(), recv_len, MPI_CHAR, src, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                std::istringstream iss2(std::string(buf.data(), recv_len));
                while (std::getline(iss2, line))
                    guesses.push_back(line);
            }

            total_guesses += m_last;
        }
        else
        {
            MPI_Send(&len, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            MPI_Send(payload.data(), len, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
        }
    }
}

vector<PT> PriorityQueue::PopMultiple(int k) {
    vector<PT> pts;
    for (int i = 0; i < k && !priority.empty(); ++i) {
        pts.push_back(priority.front());
        priority.erase(priority.begin());
    }
    return pts;
}

string serialize_pt(const PT& pt) {
    stringstream ss;
    ss << pt.pivot << " " << pt.preterm_prob << " " << pt.prob << " ";
    ss << pt.content.size() << " ";
    for (const auto& seg : pt.content) {
        ss << seg.type << " " << seg.length << " ";
    }
    ss << pt.curr_indices.size() << " ";
    for (int idx : pt.curr_indices) {
        ss << idx << " ";
    }
    ss << pt.max_indices.size() << " ";
    for (int idx : pt.max_indices) {
        ss << idx << " ";
    }
    return ss.str();
}

PT deserialize_pt(const string& str) {
    PT pt;
    stringstream ss(str);
    ss >> pt.pivot >> pt.preterm_prob >> pt.prob;
    int content_size;
    ss >> content_size;
    for (int i = 0; i < content_size; ++i) {
        int type, length;
        ss >> type >> length;
        segment seg(type, length);
        pt.content.push_back(seg);
    }
    int curr_indices_size;
    ss >> curr_indices_size;
    for (int i = 0; i < curr_indices_size; ++i) {
        int idx;
        ss >> idx;
        pt.curr_indices.push_back(idx);
    }
    int max_indices_size;
    ss >> max_indices_size;
    for (int i = 0; i < max_indices_size; ++i) {
        int idx;
        ss >> idx;
        pt.max_indices.push_back(idx);
    }
    return pt;
}

void PriorityQueue::ParallelGenerate(int k, int rank, int size) {
    vector<PT> local_pts;
    if (rank == 0) {
        local_pts = PopMultiple(k); // 根进程取出 k 个 PT
    }

    // 广播 PT 数量
    int num_pts = local_pts.size();
    MPI_Bcast(&num_pts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (num_pts == 0) return; // 没有 PT 就直接返回

    // 计算每个进程的任务范围
    int pts_per_proc = num_pts / size;
    int extra = num_pts % size;
    int start = rank * pts_per_proc + min(rank, extra);
    int end = start + pts_per_proc + (rank < extra ? 1 : 0);

    // 进程间数据分发
    if (rank == 0) {
        for (int i = 1; i < size; ++i) {
            int proc_start = i * pts_per_proc + min(i, extra);
            int proc_end = proc_start + pts_per_proc + (i < extra ? 1 : 0);
            for (int j = proc_start; j < proc_end; ++j) {
                string pt_str = serialize_pt(local_pts[j]);
                int len = pt_str.size();
                MPI_Send(&len, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                MPI_Send(pt_str.c_str(), len, MPI_CHAR, i, 0, MPI_COMM_WORLD);
            }
        }
        // 提取 rank 0 自己的部分
        local_pts = vector<PT>(local_pts.begin() + start, local_pts.begin() + end);
    } else {
        for (int j = start; j < end; ++j) {
            int len;
            MPI_Recv(&len, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            char* buf = new char[len];
            MPI_Recv(buf, len, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            string pt_str(buf, len);
            delete[] buf;
            PT pt = deserialize_pt(pt_str);
            local_pts.push_back(pt);
        }
    }

    // 每个进程处理自己的 PT，生成新的 PT
    vector<PT> local_new_pts;
    for (PT& pt : local_pts) {
        Generate(pt);
        vector<PT> temp = pt.NewPTs();
        local_new_pts.insert(local_new_pts.end(), temp.begin(), temp.end());
    }

    // 非根进程发送 new_pts 给根
    if (rank != 0) {
        int num_new = local_new_pts.size();
        MPI_Send(&num_new, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        for (PT& pt : local_new_pts) {
            string pt_str = serialize_pt(pt);
            int len = pt_str.size();
            MPI_Send(&len, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            MPI_Send(pt_str.c_str(), len, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
        }
    } else {
        // 根进程收集所有 new_pts
        vector<PT> all_new_pts = local_new_pts;

        for (int i = 1; i < size; ++i) {
            int num_new;
            MPI_Recv(&num_new, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int j = 0; j < num_new; ++j) {
                int len;
                MPI_Recv(&len, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                char* buf = new char[len];
                MPI_Recv(buf, len, MPI_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                string pt_str(buf, len);
                delete[] buf;
                PT pt = deserialize_pt(pt_str);
                all_new_pts.push_back(pt);
            }
        }

        // 对所有新 PT 计算概率并插入优先队列
        for (PT& pt : all_new_pts) {
            CalProb(pt);
            bool inserted = false;
            for (auto iter = priority.begin(); iter != priority.end(); ++iter) {
                if (pt.prob > iter->prob) {
                    priority.insert(iter, pt);
                    inserted = true;
                    break;
                }
            }
            if (!inserted) {
                priority.push_back(pt);
            }
        }
    }
}








