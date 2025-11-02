#pragma once

#include <vector>
#include "common.h"

class Engine {
        public:
                std::vector<DataPoint> dataPoint;

                void KNN(Params &p, std::vector<DataPoint> &dataset,
                                std::vector<Query> &queries);
};

