#include <vector>
#include <iostream>
#include "engine.h"
#include <algorithm>
#include <tuple>
#include <cstdlib>
#include <unordered_map>

#include <mpi.h>

double computeDistance(double* a, double* b, int dim)
{
        double sum = 0.0;
        for (int i = 0; i < dim; i++)
        {
                sum += (a[i] - b[i]) * (a[i] - b[i]);
        }
        return sum;
}

void Engine::KNN(Params& p, std::vector<DataPoint>& dataset, std::vector<Query>& queries)
{
        int numtasks, rank;
        int num_attrs, num_queries, num_data;
        MPI_Comm comm = MPI_COMM_WORLD;
        MPI_Status Stat;
        MPI_Comm_size(comm, &numtasks);
        MPI_Comm_rank(comm, &rank);

        // Broadcast parameters
        if (rank == 0)
        {
                num_attrs = p.num_attrs;
                num_queries = p.num_queries;
                num_data = p.num_data;
        }
        MPI_Bcast(&num_attrs, 1, MPI_INT, 0, comm);
        MPI_Bcast(&num_queries, 1, MPI_INT, 0, comm);
        MPI_Bcast(&num_data, 1, MPI_INT, 0, comm);

        int sendcount = num_data / numtasks;
        int recvcount = num_data / numtasks;
        if (num_data % numtasks != 0 && rank == numtasks - 1) {
                recvcount += num_data % numtasks;
        }

        std::vector<int> sendcounts;
        std::vector<int> displs;
        std::vector<int> attrs_sendcounts;
        std::vector<int> attrs_displs;
        if (rank == 0) {
                //sendcounts
                sendcounts.resize(numtasks, sendcount);
                if (num_data % numtasks != 0) {
                sendcounts[numtasks - 1] += num_data % numtasks;
                }
                //displs
                displs.resize(numtasks, 0);
                for (int i = 1; i < numtasks; i++) {
                        displs[i] = displs[i - 1] + sendcounts[i - 1];
                }
                // attrs sendcounts and displs
                attrs_sendcounts.resize(numtasks);
                attrs_displs.resize(numtasks);
                for (int i = 0; i < numtasks; i++) {
                        attrs_sendcounts[i] = sendcounts[i] * num_attrs;
                        attrs_displs[i] = displs[i] * num_attrs;
                }
        }

        // recvbuffer(s)
        std::vector<double> attrs_rx(recvcount * num_attrs);
        std::vector<int> id_rx(recvcount);
        std::vector<int> label_rx(recvcount);

        // sendbuffers(s)
        std::vector<double> attrs_tx;
        std::vector<int> id_tx;
        std::vector<int> label_tx;
        if (rank == 0) {
                id_tx.resize(num_data);
                label_tx.resize(num_data);
                attrs_tx.resize(num_data * num_attrs);

                // Init send buffers
                for (int i = 0; i < num_data; i++) {
                        id_tx[i] = dataset[i].id;
                        label_tx[i] = dataset[i].label;
                        for (int j = 0; j < num_attrs; j++) {
                                attrs_tx[i * num_attrs + j] = dataset[i].attrs[j];
                        }
                }
        }

        /*Build datatype describing structure*/
        struct tuple
        {
                double distance;
                int label;
                int id;
        };
        tuple example_tuple = { 0.0, 0, 0 };
        MPI_Datatype tuple_type;
        MPI_Datatype types[3] = { MPI_DOUBLE, MPI_INT, MPI_INT };
        int blocklengths[3] = { 1, 1, 1 };
        MPI_Aint disp[3];

        MPI_Get_address(&example_tuple.distance, &disp[0]);
        MPI_Get_address(&example_tuple.label, &disp[1]);
        MPI_Get_address(&example_tuple.id, &disp[2]);

        MPI_Aint base = disp[0];
        for (int i = 0; i < 3; i++)
        {
                disp[i] -= base;
        }

        MPI_Type_create_struct(3, blocklengths, disp, types, &tuple_type);
        MPI_Type_commit(&tuple_type);
        /* End build datatype */

        for (int i = 0; i < num_queries; i++) {
                // Bcast query
                std::vector<double> query_attrs(num_attrs);
                int query_id;
                int query_k;
                if (rank == 0) {
                        query_id = queries[i].id;
                        query_k = queries[i].k;
                        std::copy(queries[i].attrs.begin(), queries[i].attrs.end(), query_attrs.begin());
                }
                MPI_Bcast(&query_id, 1, MPI_INT, 0, comm);
                MPI_Bcast(&query_k, 1, MPI_INT, 0, comm);
                MPI_Bcast(query_attrs.data(), num_attrs, MPI_DOUBLE, 0, comm);

                // Scatter datapoints
                MPI_Scatterv(id_tx.data(), sendcounts.data(), displs.data(), MPI_INT, id_rx.data(), recvcount, MPI_INT, 0, comm);
                MPI_Scatterv(label_tx.data(), sendcounts.data(), displs.data(), MPI_INT, label_rx.data(), recvcount, MPI_INT, 0, comm);
                MPI_Scatterv(attrs_tx.data(), attrs_sendcounts.data(), attrs_displs.data(), MPI_DOUBLE,
                        attrs_rx.data(), recvcount * num_attrs, MPI_DOUBLE, 0, comm);

                // Compute local results
                std::vector<tuple> local_results; // distance, label, id
                for (int j = 0; j < recvcount; j++) {
                        double dist = computeDistance(query_attrs.data(), attrs_rx.data() + j * num_attrs, num_attrs);
                        local_results.push_back({ dist, label_rx[j], id_rx[j] });
                }
                std::sort(local_results.begin(), local_results.end(), [](const tuple& a, const tuple& b)
                        {
                                if (a.distance == b.distance)
                                {
                                        return a.label > b.label; // larger label first
                                }
                                return a.distance < b.distance; // smaller distance first
                        });

                // Master gather local results
                std::vector<tuple> best_local_results; // distance, label, id
                if (rank == 0) {
                        best_local_results.resize(numtasks * query_k);
                        std::vector<int> gather_recvcounts(numtasks, query_k);
                        std::vector<int> gather_displs(numtasks);
                        for (int i = 0; i < numtasks; i++) {
                                gather_displs[i] = i * query_k;
                        }
                        MPI_Gatherv(local_results.data(), query_k, tuple_type, best_local_results.data(), gather_recvcounts.data(), gather_displs.data(), tuple_type, 0, comm);
                } else {
                        MPI_Gatherv(local_results.data(), query_k, tuple_type, nullptr, nullptr, nullptr, tuple_type, 0, comm);
                }

                if (rank == 0) {
                        std::sort(best_local_results.begin(), best_local_results.end(), [](const tuple& a, const tuple& b)
                                {
                                        if (a.distance == b.distance)
                                                return a.label > b.label; // larger label first
                                        return a.distance < b.distance;   // smaller distance first
                                });
                        // Collect labels of top k
                        std::unordered_map<int, int> label_count;
                        std::vector<std::pair<double, int>> knn_results; // distance, id
                        for (int i = 0; i < query_k; i++)
                        {
                                knn_results.push_back(std::make_pair(best_local_results[i].distance, best_local_results[i].id));
                                label_count[best_local_results[i].label]++;
                        }

                        // Determine most frequent label
                        int max_count = 0;
                        int most_frequent_label = -1;
                        for (auto& pair : label_count)
                        {
                                if (pair.second > max_count)
                                {
                                        max_count = pair.second;
                                        most_frequent_label = pair.first;
                                }
                        }

                        // sort top k results 
                        std::sort(knn_results.begin(), knn_results.end(), [](const std::pair<double, int>& a, const std::pair<double, int>& b)
                                {
                                        if (a.first == b.first)
                                                return a.second > b.second; // larger id first
                                        return a.first < b.first;           // smaller distance first
                                });
                        reportResult(queries[i], knn_results, most_frequent_label);
                }
        }
}
