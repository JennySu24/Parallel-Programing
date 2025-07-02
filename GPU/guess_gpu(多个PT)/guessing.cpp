#include "PCFG.h"
#include <cstdint>
#include <cuda_runtime.h>
#include <vector>
#include <cstring>
#include <algorithm>
using namespace std;

void PriorityQueue::CalProb(PT &pt) {
    pt.prob = pt.preterm_prob;
    int index = 0;

    for (int idx : pt.curr_indices) {
        if (pt.content[index].type == 1) {
            pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq;
        }
        if (pt.content[index].type == 2) {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
        }
        if (pt.content[index].type == 3) {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
        }
        index += 1;
    }
}

void PriorityQueue::init() {
    for (PT pt : m.ordered_pts) {
        for (segment seg : pt.content) {
            if (seg.type == 1) {
                pt.max_indices.emplace_back(m.letters[m.FindLetter(seg)].ordered_values.size());
            }
            if (seg.type == 2) {
                pt.max_indices.emplace_back(m.digits[m.FindDigit(seg)].ordered_values.size());
            }
            if (seg.type == 3) {
                pt.max_indices.emplace_back(m.symbols[m.FindSymbol(seg)].ordered_values.size());
            }
        }
        pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;
        CalProb(pt);
        priority.push_back(pt);
    }
    std::make_heap(priority.begin(), priority.end(), PTComparator());
}

vector<PT> PT::NewPTs() {
    vector<PT> res;
    if (content.size() == 1) {
        return res;
    }
    else {
        int init_pivot = pivot;
        for (int i = pivot; i < curr_indices.size() - 1; i += 1) {
            curr_indices[i] += 1;
            if (curr_indices[i] < max_indices[i]) {
                pivot = i;
                res.emplace_back(*this);
            }
            curr_indices[i] -= 1;
        }
        pivot = init_pivot;
        return res;
    }
}

void PriorityQueue::PopNext() {
    size_t batch_size = 8;
    if (priority.empty()) return;

    vector<PT> pts_to_generate;
    vector<PT> new_pts_to_add;

    // 取出 batch_size 个 PT 进行批量处理
    for (size_t i = 0; i < batch_size && !priority.empty(); ++i) {
        std::pop_heap(priority.begin(), priority.end(), PTComparator());
        PT pt = priority.back();
        priority.pop_back();

        pts_to_generate.push_back(pt);
        vector<PT> new_pts = pt.NewPTs();
        new_pts_to_add.insert(new_pts_to_add.end(), new_pts.begin(), new_pts.end());
    }

    // 批量生成 guesses
    vector<string> batch_guesses;
    Generate_GPU(pts_to_generate, batch_guesses);
    guesses.insert(guesses.end(), batch_guesses.begin(), batch_guesses.end());

    // 批量添加新的 PT
    for (PT& new_pt : new_pts_to_add) {
        CalProb(new_pt);
        priority.push_back(new_pt);
        std::push_heap(priority.begin(), priority.end(), PTComparator());
    }
}

void PriorityQueue::Generate(const PT& pt, vector<string>& local_guesses) {
    CalProb(const_cast<PT&>(pt));
    if (pt.content.size() == 1) {
        segment *a = nullptr;
        if (pt.content[0].type == 1) a = &m.letters[m.FindLetter(pt.content[0])];
        else if (pt.content[0].type == 2) a = &m.digits[m.FindDigit(pt.content[0])];
        else if (pt.content[0].type == 3) a = &m.symbols[m.FindSymbol(pt.content[0])];
        for (int i = 0; i < pt.max_indices[0]; i++)
            local_guesses.emplace_back(a->ordered_values[i]);
        total_guesses += pt.max_indices[0];
    }
    else {
        string guess;
        int seg_idx = 0;
        for (int idx : pt.curr_indices) {
            if (pt.content[seg_idx].type == 1)
                guess += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            else if (pt.content[seg_idx].type == 2)
                guess += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            else if (pt.content[seg_idx].type == 3)
                guess += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            seg_idx++;
            if (seg_idx == pt.content.size() - 1) break;
        }
        segment *a = nullptr;
        if (pt.content.back().type == 1) a = &m.letters[m.FindLetter(pt.content.back())];
        else if (pt.content.back().type == 2) a = &m.digits[m.FindDigit(pt.content.back())];
        else if (pt.content.back().type == 3) a = &m.symbols[m.FindSymbol(pt.content.back())];
        for (int i = 0; i < pt.max_indices.back(); i++)
            local_guesses.emplace_back(guess + a->ordered_values[i]);
        total_guesses += pt.max_indices.back();
    }
}








