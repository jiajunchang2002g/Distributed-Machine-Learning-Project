#pragma once

#include <vector>
#include <tuple>
#include <string>
#include <queue>
#include "common.h"

// Comparator for max-heap
struct CompareDist {
        bool operator()(const std::tuple<double, int, int> &a,
                        const std::tuple<double, int, int> &b) const {
                // larger distance = higher priority
                if (std::get<0>(a) != std::get<0>(b))
                        return std::get<0>(a) < std::get<0>(b);
                // tie-break by label (smaller label wins)
                return std::get<1>(a) > std::get<1>(b);
        }
};

class Engine {
        public:
                std::vector<DataPoint> dataPoint;

                double computeDistance(const std::vector<double> &a,
                                const std::vector<double> &b);

                void KNN(Params &p, std::vector<DataPoint> &dataset,
                                std::vector<Query> &queries);
};

