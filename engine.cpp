#include <vector>
#include <iostream>
#include "engine.h"
#include <algorithm>
#include <tuple>
#include <cstdlib>
#include <unordered_map>

#include <mpi.h>

double computeDistance(double *a, double *b, int dim)
{
        double sum = 0.0;
        for (int i = 0; i < dim; i++)
        {
                sum += (a[i] - b[i]) * (a[i] - b[i]);
        }
        return sum;
}

void Engine::KNN(Params &p, std::vector<DataPoint> &dataset, std::vector<Query> &queries) {
        int numtasks, rank;
        int num_attrs, num_queries, num_data, sendcount, recvcount;
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
        
        sendcount = num_data / numtasks;
        recvcount = num_data / numtasks;
        
        // recvbuffer(s)
        std::vector<double> attrs_rx(recvcount * num_attrs);
        std::vector<int> id_rx(recvcount);
        std::vector<int> label_rx(recvcount);
        
        // sendbuffers(s)
        std::vector<double> attrs_tx;
        std::vector<int> id_tx;
        std::vector<int> label_tx;
        if (rank == 0)
        {
                id_tx.resize(num_data);
                label_tx.resize(num_data);
                attrs_tx.resize(num_data * num_attrs);

                // Init send buffers
                for (int i = 0; i < num_data; i++)
                {
                        id_tx[i] = dataset[i].id;
                        label_tx[i] = dataset[i].label;
                        for (int j = 0; j < num_attrs; j++)
                        {
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
        tuple example_tuple = {0.0, 0, 0};
        MPI_Datatype tuple_type;
        MPI_Datatype types[3] = {MPI_DOUBLE, MPI_INT, MPI_INT};
        int blocklengths[3] = {1, 1, 1};
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
        
        for (int i = 0; i < num_queries; i++)
        {
                int query_id;
                int query_k;
                std::vector<double> query_attrs(num_attrs);
                if (rank == 0)
                {
                        query_id = queries[i].id;
                        query_k = queries[i].k;
                        // copy attributes
                        for (int j = 0; j < num_attrs; j++)
                        {
                                query_attrs[j] = queries[i].attrs[j];
                        }
                }
                MPI_Bcast(&query_id, 1, MPI_INT, 0, comm);
                MPI_Bcast(&query_k, 1, MPI_INT, 0, comm);
                MPI_Bcast(query_attrs.data(), num_attrs, MPI_DOUBLE, 0, comm);

                MPI_Scatter(id_tx.data(), sendcount, MPI_INT, id_rx.data(), recvcount, MPI_INT, 0, comm);
                MPI_Scatter(label_tx.data(), sendcount, MPI_INT, label_rx.data(), recvcount, MPI_INT, 0, comm);
                MPI_Scatter(attrs_tx.data(), sendcount * num_attrs, MPI_DOUBLE,
                        attrs_rx.data(), recvcount * num_attrs, MPI_DOUBLE, 0, comm);

                std::vector<tuple> local_results; // distance, label, id
                        for (int j = 0; j < recvcount; j++)
                        {
                                double dist = computeDistance(query_attrs.data(), attrs_rx.data() + j * num_attrs, num_attrs);
                                local_results.push_back({dist, label_rx[j], id_rx[j]});
                        }
                        std::sort(local_results.begin(), local_results.end(), [](const tuple &a, const tuple &b)
                        {
                                if (a.distance == b.distance) {
                                        return a.label > b.label; // larger label first
                                }
                                return a.distance < b.distance;   // smaller distance first
                        });
                        std::vector<std::pair<double, int>> knn_results; // distance, id
                        
                        std::vector<tuple> best_local_results;
                        if (rank == 0)
                        {
                                best_local_results.resize(numtasks * query_k);
                        }
                        std::cout << "Rank " << rank << " starting gathering for query " << query_id << std::endl;
                        
                        MPI_Gather(local_results.data(), query_k, tuple_type,
                        best_local_results.data(), query_k, tuple_type,
                        0, comm);
                        
                        // std::cout << "Rank " << rank << " finished gathering for query " << query_id << std::endl;
                        // print best local results
                        if (rank == 0)
                        {
                                std::cout << "Best local results for query " << query_id << ":" << std::endl;
                                for (int j = 0; j < numtasks * query_k; j++)
                                {
                                        std::cout << "Distance: " << best_local_results[j].distance
                                        << ", Label: " << best_local_results[j].label
                                        << ", ID: " << best_local_results[j].id << std::endl;
                                }
                        }
                        if (rank == 0)
                        {
                                std::sort(best_local_results.begin(), best_local_results.end(), [](const tuple &a, const tuple &b)
                                {
                                        if (a.distance == b.distance)
                                        return a.label > b.label; // larger label first
                                        return a.distance < b.distance;   // smaller distance first
                                });
                                int most_frequent_label = -1;
                                std::unordered_map<int, int> label_count;
                                for (int i = 0; i < query_k; i++)
                                {
                                        knn_results.push_back(std::make_pair(best_local_results[i].distance, best_local_results[i].id));
                                        label_count[best_local_results[i].label]++;
                                }
                                
                                int max_count = 0;
                                for (auto &pair : label_count)
                                {
                                        if (pair.second > max_count)
                                        {
                                                max_count = pair.second;
                                                most_frequent_label = pair.first;
                                        }
                                }
                                
                                std::sort(knn_results.begin(), knn_results.end(), [](const std::pair<double, int> &a, const std::pair<double, int> &b)
                                {
                                        if (a.first == b.first)
                                        return a.second > b.second; // larger id first
                                        return a.first < b.first;           // smaller distance first
                                });
                                reportResult(queries[i], knn_results, most_frequent_label);
                        }
                }
        }
        