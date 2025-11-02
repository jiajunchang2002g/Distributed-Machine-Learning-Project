#include <vector>
#include <unordered_map>
#include <queue>
#include <algorithm>
#include <iostream>
#include <cmath>
#include "engine.h"

#include <mpi.h>

using namespace std;


void Engine::KNN(Params &p, vector<DataPoint> &dataset, vector<Query> &queries) {

        int numtasks, rank, dest, src, rc, count, tag=1;
        int sendcount, recvcount;
        MPI_Comm comm = MPI_COMM_WORLD;
        MPI_Status Stat;
        MPI_Comm_size(comm, &numtasks);
        MPI_Comm_rank(comm, &rank);

        sendcount = p.num_data / numtasks;
        recvcount = p.num_data / numtasks;

        std::vector<int> id_in(recvcount); 
        std::vector<int> label_in(recvcount);
        std::vector<std::vector<double>> attrs_in(recvcount);
        std::vector<int> id_out;
        std::vector<int> label_out;
        std::vector<std::vector<double>> attrs_out;

        if (0 == rank) {
                id_out.resize(p.num_data); 
                label_out.resize(p.num_data); 
                attrs_out.resize(p.num_data);
                for (int i = 0; i < p.num_data; i++) {
                        id_out[i] = dataset[i].id;
                        label_out[i] = dataset[i].label;
                        attrs_out[i].resize(p.num_attrs); 
                        attrs_out[i] = dataset[i].attrs;
                }
        }

        if (0 == rank) {
                MPI_Scatter(id_out.data(), sendcount, MPI_INT, id_in.data(), recvcount, MPI_INT, src, comm);
        } else {
                MPI_Scatter(nullptr, sendcount, MPI_INT, id_in.data(), recvcount, MPI_INT, src, comm);
        }
        // MPI_Scatter(attrs_out.data(), p.num_data, MPI_DOUBLE, attrs_in.data(), p.num_attrs, MPI_DOUBLE, src, comm);
        // MPI_Scatter(label_out.data(), p.num_data, MPI_INT, label_in.data(), p.num_data, MPI_INT, src, comm);

        if (0 == rank) {
                printf("rank= %d  Ids: %d %d %d %d\n",rank,id_in[0],id_in[1],id_in[2],id_in[3]);
                printf("rank= %d  Labels: %d %d %d %d\n",rank,label_in[0],label_in[1],label_in[2],label_in[3]);
        }
}
