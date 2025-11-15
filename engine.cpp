#include "engine.h"
#include "utils.h"

#include <mpi.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <tuple>
#include <unordered_map>
#include <vector>
#include <cmath>

double computeDistance(double *a, double *b, int dim) {
      double sum = 0.0;
      for (int i = 0; i < dim; i++) {
            sum += (a[i] - b[i]) * (a[i] - b[i]);
      }
      return sum;
}

void Engine::KNN(Params &p, std::vector<DataPoint> &dataset,
                 std::vector<Query> &queries) 
{
    MPI_Comm world = MPI_COMM_WORLD;
    int size, rank;
    MPI_Comm_size(world, &size);
    MPI_Comm_rank(world, &rank);

    int num_attrs, num_queries, num_data;
    if (rank == 0) {
        num_attrs   = p.num_attrs;
        num_queries = p.num_queries;
        num_data    = p.num_data;
    }
    MPI_Bcast(&num_attrs,   1, MPI_INT, 0, world);
    MPI_Bcast(&num_queries, 1, MPI_INT, 0, world);
    MPI_Bcast(&num_data,    1, MPI_INT, 0, world);

    // ============================================================
    // 1. Build 2D process grid (square grid)
    // ============================================================
    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);
    int periods[2] = {0, 0};
    MPI_Comm cart;
    MPI_Cart_create(world, 2, dims, periods, 0, &cart);

    int coords[2];
    MPI_Cart_coords(cart, rank, 2, coords);
    int row = coords[0];
    int col = coords[1];

    // Create row/column communicators
    int row_dims[2] = {true, false};  // keep row fixed
    int col_dims[2] = {false, true};  // keep col fixed

    MPI_Comm row_comm, col_comm;
    MPI_Cart_sub(cart, row_dims, &row_comm);
    MPI_Cart_sub(cart, col_dims, &col_comm);  

    // ============================================================
    // 2. Scatter datapoints from master to first row of workers
    // ============================================================
    int dp_remainder = num_data % dims[0];
    int dp_recvcount = num_data / dims[0] + (row == dims[0]-1 ? dp_remainder : 0);

    std::vector<int>    dp_id_recv_buf(dp_recvcount);
    std::vector<int>    dp_label_recv_buf(dp_recvcount);
    std::vector<double> dp_attr_recv_buf(dp_recvcount * num_attrs);

    // MASTER ONLY
    if (rank == 0) {
        std::vector<int> sendcounts(dims[0]), displs(dims[0]);
        build_sendcounts_displs(num_data, dims[0], row, sendcounts, displs);
        MPI_Scatterv(dataset.data(), sendcounts.data(), displs.data(),
                     MPI_INT, dp_id_recv_buf.data(), dp_recvcount, MPI_INT,
                     0, col_comm);
        // Scatter datapoint labels
        std::vector<int> labels(num_data);
        for (int i = 0; i < num_data; i++)
            labels[i] = dataset[i].label;
        MPI_Scatterv(labels.data(), sendcounts.data(), displs.data(),
                     MPI_INT, dp_label_recv_buf.data(), dp_recvcount, MPI_INT,
                     0, col_comm);
        // Scatter datapoint attributes
        std::vector<double> all_attrs(num_data * num_attrs);
        for (int i = 0; i < num_data; i++)
            for (int a = 0; a < num_attrs; a++)
                all_attrs[i * num_attrs + a] = dataset[i].attrs[a];
        std::vector<int> attr_sc(dims[0]), attr_disp(dims[0]);
        build_sendcounts_displs_attrs(sendcounts, num_attrs,
                                         attr_sc, attr_disp);
        MPI_Scatterv(all_attrs.data(), attr_sc.data(), attr_disp.data(),
                     MPI_DOUBLE, dp_attr_recv_buf.data(), dp_recvcount*num_attrs,
                     MPI_DOUBLE, 0, col_comm);
    } 
    
    // FIRST ROW receive from MASTER
    else if (row == 0) {
        MPI_Scatterv(nullptr, nullptr, nullptr,
                     MPI_INT, dp_id_recv_buf.data(), dp_recvcount, MPI_INT,
                     0, col_comm);

        MPI_Scatterv(nullptr, nullptr, nullptr,
                     MPI_INT, dp_label_recv_buf.data(), dp_recvcount, MPI_INT,
                     0, col_comm);

        MPI_Scatterv(nullptr, nullptr, nullptr,
                     MPI_DOUBLE, dp_attr_recv_buf.data(), dp_recvcount*num_attrs,
                     MPI_DOUBLE, 0, col_comm);
    }

    // FIRST ROW broadcast to OTHER ROWS
    MPI_Bcast(dp_id_recv_buf.data(),    dp_recvcount,         MPI_INT,    0, row_comm);
    MPI_Bcast(dp_label_recv_buf.data(), dp_recvcount,         MPI_INT,    0, row_comm);
    MPI_Bcast(dp_attr_recv_buf.data(),  dp_recvcount*num_attrs, MPI_DOUBLE, 0, row_comm);

    // ============================================================
    // 3. Partition queries along first columns
    // ============================================================
    int q_remainder = num_queries % dims[1];
    int q_recvcount = num_queries / dims[1] + (col == dims[1]-1 ? q_remainder : 0);

    std::vector<Query> local_queries(q_recvcount);
    std::vector<int> query_id_local(q_recvcount);
    std::vector<int> query_k_local(q_recvcount);
    std::vector<double> query_attr_local(q_recvcount * num_attrs);

    // MASTER ONLY scatter queries to COLUMN ROOTS
    if (rank == 0) {
        std::vector<int> q_sendcounts(dims[1]), q_displs(dims[1]);
        build_sendcounts_displs(num_queries, dims[1], col,
                                    q_sendcounts, q_displs);
        std::vector<int> q_ids(num_queries), q_ks(num_queries);
        MPI_Scatterv(q_ids.data(), q_sendcounts.data(), q_displs.data(),
                     MPI_INT, query_id_local.data(), q_recvcount, MPI_INT,
                     0, row_comm);
        MPI_Scatterv(q_ks.data(), q_sendcounts.data(), q_displs.data(),
                     MPI_INT, query_k_local.data(), q_recvcount, MPI_INT,
                     0, row_comm);
        std::vector<double> q_attrs(num_queries * num_attrs);
        for (int i = 0; i < num_queries; i++) {
            q_ids[i] = queries[i].id;
            q_ks[i]  = queries[i].k;
            for (int a = 0; a < num_attrs; a++)
                q_attrs[i * num_attrs + a] = queries[i].attrs[a];
        }
        std::vector<int> q_attrs_sc(dims[1]), q_attrs_disp(dims[1]);
        build_sendcounts_displs_attrs(q_sendcounts, num_attrs,
                                         q_attrs_sc, q_attrs_disp);
        std::vector<double> query_attr_local(q_recvcount * num_attrs);
        MPI_Scatterv(q_attrs.data(), q_attrs_sc.data(), q_attrs_disp.data(),
                     MPI_DOUBLE, query_attr_local.data(), q_recvcount*num_attrs,
                     MPI_DOUBLE, 0, row_comm);

        for (int i = 0; i < q_recvcount; i++) {
            local_queries[i].id = query_id_local[i];
            local_queries[i].k  = query_k_local[i];
            local_queries[i].attrs.resize(num_attrs);
            for (int a = 0; a < num_attrs; a++)
                local_queries[i].attrs[a] = query_attr_local[i * num_attrs + a];
        }
    }
    // COLUMN ROOTS receive from MASTER
    else if (col == 0) {
        MPI_Scatterv(nullptr, nullptr, nullptr,
                     MPI_INT, query_id_local.data(), q_recvcount, MPI_INT,
                     0, row_comm);

        MPI_Scatterv(nullptr, nullptr, nullptr,
                     MPI_INT, query_k_local.data(), q_recvcount, MPI_INT,
                     0, row_comm);

        std::vector<double> query_attr_local(q_recvcount * num_attrs);
        MPI_Scatterv(nullptr, nullptr, nullptr, MPI_DOUBLE,
                     query_attr_local.data(), q_recvcount*num_attrs,
                     MPI_DOUBLE, 0, row_comm);

        for (int i = 0; i < q_recvcount; i++) {
            local_queries[i].id = query_id_local[i];
            local_queries[i].k  = query_k_local[i];
            local_queries[i].attrs.resize(num_attrs);
            for (int a = 0; a < num_attrs; a++)
                local_queries[i].attrs[a] = query_attr_local[i * num_attrs + a];
        }
    }

    // Broadcast queries down each row
    for (int i = 0; i < q_recvcount; i++) {
        MPI_Bcast(&local_queries[i].id, 1, MPI_INT, 0, col_comm);
        MPI_Bcast(&local_queries[i].k,  1, MPI_INT, 0, col_comm);
        local_queries[i].attrs.resize(num_attrs);
        MPI_Bcast(local_queries[i].attrs.data(), num_attrs, MPI_DOUBLE, 0, col_comm);
    }

    // ============================================================
    // 4. Define tuple type for (distance, label, id)
    // ============================================================
    struct tuple { double distance; int label; int id; };
    tuple example;
    MPI_Datatype tuple_type;
    MPI_Datatype types[3] = {MPI_DOUBLE, MPI_INT, MPI_INT};
    int blocklengths[3]  = {1,1,1};
    MPI_Aint disp[3];

    MPI_Get_address(&example.distance, &disp[0]);
    MPI_Get_address(&example.label,    &disp[1]);
    MPI_Get_address(&example.id,       &disp[2]);
    MPI_Aint base = disp[0];
    for (int i = 0; i < 3; i++) disp[i] -= base;

    MPI_Type_create_struct(3, blocklengths, disp, types, &tuple_type);
    MPI_Type_commit(&tuple_type);

    // ============================================================
    // 5. Local computation 
    // ============================================================
    std::vector<std::vector<tuple>> local_results(q_recvcount);
    
    for (int qi = 0; qi < q_recvcount; qi++) {

        // for each query, compute distances to all local datapoints
        local_results[qi].clear();
        int k = local_queries.at(qi).k;
        for (int dp = 0; dp < dp_recvcount; dp++) {
            double dist = computeDistance(
                local_queries.at(qi).attrs.data(),
                &dp_attr_recv_buf[dp * num_attrs],
                num_attrs
            );
            local_results[qi].push_back({dist, dp_label_recv_buf.at(dp), dp_id_recv_buf.at(dp)});
        }

        // keep only top-k closest
        std::nth_element(local_results[qi].begin(),
                         local_results[qi].begin() + k,
                         local_results[qi].end(),
                         [](const tuple &a, const tuple &b){
                             if (a.distance == b.distance)
                                 return a.label > b.label;
                             return a.distance < b.distance;
                         });
        local_results[qi].resize(k);
    }

    std::vector<tuple> local_results_flat;
    std::vector<int> k_per_query(q_recvcount);
    for (int qi = 0; qi < q_recvcount; qi++) {
        int k = local_results[qi].size(); 
        k_per_query[qi] = k;
        local_results_flat.insert(local_results_flat.end(),
                        local_results[qi].begin(),
                        local_results[qi].end());
    }

    // ============================================================
    // 6. Gather values from other columns to column roots
    // ============================================================
    std::vector<std::vector<tuple>> gathered_results;
    if (col == 0) {
        std::vector<tuple> results_recv_buf(local_results_flat.size() * dims[0]);
        std::vector<int> sendcounts(dims[0]);
        std::vector<int> displs(dims[0]);
        build_sendcounts_displs(local_results_flat.size() * dims[0], dims[0], row, sendcounts, displs);
        MPI_Gatherv(local_results_flat.data(), local_results_flat.size(), tuple_type,
                    results_recv_buf.data(), sendcounts.data(), displs.data(),
                    tuple_type, 0, col_comm);
        // Assemble vector of results
        gathered_results.resize(q_recvcount);
        for (int qi = 0; qi < q_recvcount; qi++) {
            gathered_results[qi].clear();
            for (int c = 0; c < dims[0]; c++) {
                for (int i = 0; i < k_per_query[qi]; i++) {
                    int index = c * local_results_flat.size() + qi * k_per_query[qi] + i;
                    gathered_results[qi].push_back(results_recv_buf.at(index));
                }
            }
        }
    } 
    else {
        MPI_Gatherv(local_results_flat.data(), local_results_flat.size(), tuple_type,
                    nullptr, nullptr, nullptr,
                    tuple_type, 0, col_comm);
    }

    if (col == 0) {
        for (int qi = 0; qi < q_recvcount; qi++) {
            int k = local_queries.at(qi).k;
            // keep only top-k closest
            std::nth_element(gathered_results[qi].begin(),
                             gathered_results[qi].begin() + k,
                             gathered_results[qi].end(),
                             [](const tuple &a, const tuple &b){
                                 if (a.distance == b.distance)
                                     return a.label > b.label;
                                 return a.distance < b.distance;
                             });
            gathered_results[qi].resize(k);
        }
    }

    // ============================================================
    // 7. Gather final results from column roots to rank 0
    // ============================================================
    std::vector<tuple> final_results_recv_buf;
    if (rank == 0) {
        std::vector<tuple> final_results_recv_buf(dims[1] * gathered_results[0].size());
        std::vector<int> sendcounts(dims[1]);
        std::vector<int> displs(dims[1]);
        build_sendcounts_displs(dims[1] * gathered_results[0].size(), dims[1], 0, sendcounts, displs);
        MPI_Gatherv(gathered_results[0].data(), gathered_results[0].size(), tuple_type,
                    final_results_recv_buf.data(), sendcounts.data(), displs.data(),
                    tuple_type, 0, row_comm);
    } 
    else if (col == 0) {
        std::vector<tuple> send_buf;
        for (int qi = 0; qi < q_recvcount; qi++) {
            send_buf.insert(send_buf.end(),
                            gathered_results[qi].begin(),
                            gathered_results[qi].end());
        }
        MPI_Gatherv(send_buf.data(), send_buf.size(), tuple_type,
                    nullptr, nullptr, nullptr,
                    tuple_type, 0, row_comm);
    }

    // ============================================================
    // 8. Rank 0 reconstructs full results and prints
    // ============================================================
    if (rank == 0) {
        // unpack final results buffer
        std::unordered_map<int, std::vector<tuple>> results_map;
        for (int c = 0; c < dims[1]; c++) {
            for (int i = 0; i < gathered_results[0].size(); i++) {
                int index = c * gathered_results[0].size() + i; 
                int query_id = final_results_recv_buf.at(index).id;
                results_map[query_id].push_back(final_results_recv_buf.at(index));
            }
        }
        // print results in order of query IDs
        for (int qi = 0; qi < num_queries; qi++) {
            auto &res = results_map[qi];
            std::unordered_map<int, int> label_count;
            for (auto &tup : res) {
                label_count[tup.label]++;
            }
            int max_count = 0;
            int predicted_label = -1;
            for (auto &lc : label_count) {
                if (lc.second > max_count || (lc.second == max_count && lc.first > predicted_label)) {
                    max_count = lc.second;
                    predicted_label = lc.first; 
                }
            }
            std::sort(res.begin(), res.end(),
                      [](const tuple &a, const tuple &b){
                          if (a.distance == b.distance)
                              return a.id > b.id;
                          return a.distance < b.distance;
                      });
            // convert tuple to pair
            std::vector<std::pair<double, int>> res_pairs;
            for (const auto &tup : res) {
                res_pairs.emplace_back(tup.distance, tup.label);
            }
            reportResult(queries[qi], res_pairs, predicted_label);
        }
    }

    MPI_Type_free(&tuple_type);
}
