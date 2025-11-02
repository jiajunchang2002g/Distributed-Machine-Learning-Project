#include <vector>
#include <iostream>
#include "engine.h"
#include <algorithm>
#include <tuple>
#include <cstdlib>
#include <unordered_map>

#include <mpi.h>

using namespace std;

double computeDistance(double *a, double *b, int dim)
{
        double sum = 0.0;
        for (int i = 0; i < dim; i++)
        {
                sum += (a[i] - b[i]) * (a[i] - b[i]);
        }
        return sum;
}

void Engine::KNN(Params &p, vector<DataPoint> &dataset, vector<Query> &queries)
{
        int numtasks, rank, dest, src, rc, count, tag = 1;
        int num_attrs, num_queries, num_data, sendcount, recvcount;
        MPI_Comm comm = MPI_COMM_WORLD;
        MPI_Status Stat;
        MPI_Comm_size(comm, &numtasks);
        MPI_Comm_rank(comm, &rank);

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

        cout << "Rank " << rank << " will sendcount: " << sendcount << ", recvcount: " << recvcount << endl;
        cout.flush();
        
        // recvbuffer(s)
        double **attrs_rx = (double **)malloc(sizeof(double *) * recvcount);
        for (int i = 0; i < recvcount; i++)
        {
                attrs_rx[i] = (double *)malloc(sizeof(double) * num_attrs);
        }
        int *id_rx = (int *)malloc(sizeof(int) * recvcount);
        int *label_rx = (int *)malloc(sizeof(int) * recvcount);
        double **q_attrs_rx = (double **)malloc(sizeof(double *) * num_attrs);

        // sendbuffers(s)
        double **attrs_tx;
        int *id_tx;
        int *label_tx;
        if (rank == 0)
        {
                attrs_tx = (double **)malloc(sizeof(double *) * sendcount);
                for (int i = 0; i < sendcount; i++)
                {
                        attrs_tx[i] = (double *)malloc(sizeof(double) * num_attrs);
                }
                id_tx = (int *)malloc(sizeof(int) * sendcount);
                label_tx = (int *)malloc(sizeof(int) * sendcount);

                // prepare send buffers
                for (int i = 0; i < sendcount; i++)
                {
                        id_tx[i] = dataset[i].id;
                        label_tx[i] = dataset[i].label;
                        attrs_tx[i] = dataset[i].attrs.data();
                }
        }

        std::cout << "Rank " << rank << " starting KNN with " << num_queries << " queries." << endl;
        std::cout.flush();
        for (int i = 0; i < num_queries; i++)
        {
                int query_id;
                int query_k;
                double *query_attrs = (double *)malloc(sizeof(double) * num_attrs);
                if (rank == 0)
                {
                        query_id = queries[i].id;
                        query_k = queries[i].k;
                        query_attrs = queries[i].attrs.data();
                }
                MPI_Bcast(&query_id, 1, MPI_INT, 0, comm);
                MPI_Bcast(&query_k, 1, MPI_INT, 0, comm);
                MPI_Bcast(query_attrs, num_attrs, MPI_DOUBLE, 0, comm);

                MPI_Scatter(id_tx, sendcount, MPI_INT, id_rx, recvcount, MPI_INT, 0, comm);
                MPI_Scatter(label_tx, sendcount, MPI_INT, label_rx, recvcount, MPI_INT, 0, comm);
                MPI_Scatter(attrs_tx[0], sendcount * num_attrs, MPI_DOUBLE,
                            attrs_rx[0], recvcount * num_attrs, MPI_DOUBLE, 0, comm);

                std::vector<std::tuple<double, int, int>> local_results; // distance, label, id
                for (int j = 0; j < recvcount; j++)
                {
                        double dist = computeDistance(query_attrs, attrs_rx[j], num_attrs);
                        local_results.push_back(std::make_tuple(dist, label_rx[j], id_rx[j]));
                }
                std::sort(local_results.begin(), local_results.end());
                std::vector<std::pair<double, int>> knn_results; // distance, id

                /*Build datatype describing structure*/
                MPI_Datatype tuple_type;
                MPI_Datatype types[3] = {MPI_DOUBLE, MPI_INT, MPI_INT};
                int blocklengths[3] = {1, 1, 1};
                // manually compute displacements
                MPI_Aint disp[3];
                MPI_Aint base, sizeofentry;

                // Compute displacements
                MPI_Get_address(&std::get<0>(local_results[0]), &disp[0]);
                MPI_Get_address(&std::get<1>(local_results[0]), &disp[1]);
                MPI_Get_address(&std::get<2>(local_results[0]), &disp[2]);
                base = disp[0];
                for (i=0; i < 3; i++) disp[i] = MPI_Aint_diff(disp[i], base);

                MPI_Type_create_struct(3, blocklengths, disp, types, &tuple_type);

                /*Compute extent of the structure*/
                MPI_Get_address(local_results.data() + 1, &sizeofentry);
                sizeofentry = MPI_Aint_diff(sizeofentry, base);

                /*Build datatype describing structure*/
                MPI_Type_create_resized(tuple_type, 0, sizeofentry, &tuple_type);
                MPI_Type_commit(&tuple_type);

                std::vector<std::tuple<double, int, int>> best_local_results;
                if (rank == 0)
                {
                        best_local_results.resize(numtasks * query_k);
                }
                MPI_Gather(&local_results[0], query_k * sizeof(std::tuple<double, int, int>), tuple_type,
                           &best_local_results[0], query_k * sizeof(std::tuple<double, int, int>), tuple_type,
                           0, comm);
                MPI_Type_free(&tuple_type);
                std::cout << "Rank " << rank << " finished gathering for query " << query_id << endl;
                // print best local results
                if (rank == 0)
                {
                        std::cout << "Best local results for query " << query_id << ":" << std::endl;
                        for (int j = 0; j < numtasks * query_k; j++)
                        {
                                std::cout << "Distance: " << std::get<0>(best_local_results[j])
                                          << ", Label: " << std::get<1>(best_local_results[j])
                                          << ", ID: " << std::get<2>(best_local_results[j]) << std::endl;
                        }
                }
                std::cout.flush();
                if (rank == 0)
                {
                        std::sort(best_local_results.begin(), best_local_results.end());
                        int most_frequent_label = -1;
                        std::unordered_map<int, int> label_count;
                        for (int i = 0; i < query_k; i++)
                        {
                                knn_results.push_back(std::make_pair(std::get<0>(best_local_results[i]), std::get<2>(best_local_results[i])));
                                label_count[std::get<1>(best_local_results[i])]++;
                        }
                        // pick most frequent label
                        int max_count = 0;
                        for (auto &pair : label_count)
                        {
                                if (pair.second > max_count)
                                {
                                        max_count = pair.second;
                                        most_frequent_label = pair.first;
                                }
                        }
                        // sort results
                        sort(knn_results.begin(), knn_results.end());
                        reportResult(queries[i], knn_results, most_frequent_label);
                }
        }
}
