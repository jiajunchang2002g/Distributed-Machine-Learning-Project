#include <vector>
#include <tuple>
#include <unordered_map>
#include <queue>
#include <algorithm>
#include <iostream>
#include <cmath>
#include "engine.h"

using namespace std;

double Engine::computeDistance(const vector<double> &pointA,
                const vector<double> &pointB) {
        double sum = 0.0;
        for (size_t i = 0; i < pointA.size(); ++i) {
                double diff = pointA[i] - pointB[i];
                sum += diff * diff;
        }
        return sum;
}

void Engine::KNN(Params &p, vector<DataPoint> &dataset, vector<Query> &queries) {
        for (auto &query : queries) {

                // 1. Max-heap of (dist, label, id)
                priority_queue<tuple<double, int, int>,
                vector<tuple<double, int, int>>,
                CompareDist> heap;

                for (size_t i = 0; i < dataset.size(); ++i) {
                        double dist = computeDistance(dataset[i].attrs, query.attrs);
                        int label = dataset[i].label;
                        int id = static_cast<int>(i);

                        if (heap.size() < query.k) {
                                heap.push({dist, label, id});
                        } else if (dist < get<0>(heap.top()) ||
                                        (dist == get<0>(heap.top()) && label > get<1>(heap.top()))) {
                                heap.pop();
                                heap.push({dist, label, id});
                        }
                }

                vector<std::pair<double, int>> result;  
                while (!heap.empty()) {
                        auto t = heap.top();
                        result.push_back({get<0>(t), get<2>(t)}); 
                        heap.pop();
                }

                // 3. Sort result by id
                sort(result.begin(), result.end(),
                                [](const pair<double,int> &a, const pair<double,int> &b) {
                                return a.second < b.second;
                                });

                // 4. Count label frequencies to assign query label
                unordered_map<int,int> freqs;
                for (auto &t : result) {  
                        int label = get<1>(t);
                        freqs[label]++;
                }

                // 5. Manually pick most frequent label
                int best_label = -1;
                int best_count = -1;
                for (auto &entry : freqs) {
                        if (entry.second > best_count) {
                                best_count = entry.second;
                                best_label = entry.first;
                        } else if (entry.second == best_count && entry.first < best_label) {
                                best_label = entry.first;
                        }
                }

                // 6. Report results
                reportResult(query, result, best_label);
        }
}

