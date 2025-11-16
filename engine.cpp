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
                std::vector<Query> &queries) {
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
        int dp_recvcount = num_data / dims[0] + (row == 0 ? dp_remainder : 0);  // root handles remainder

        std::vector<int>    dp_id_recv_buf(dp_recvcount);
        std::vector<int>    dp_label_recv_buf(dp_recvcount);
        std::vector<double> dp_attr_recv_buf(dp_recvcount * num_attrs);

        // MASTER ONLY
        if (rank == 0) {
                std::vector<int> sendcounts(dims[0]), displs(dims[0]);

                // All first row ranks tell master their recvcounts
                MPI_Gather(&dp_recvcount, 1, MPI_INT, sendcounts.data(), 1, MPI_INT, 0, col_comm);
                // Build displacements
                displs[0] = 0;
                for (int i = 1; i < dims[0]; i++)
                        displs[i] = displs[i-1] + sendcounts[i-1];

                // Scatter datapoint IDs
                std::vector<int> dp_ids(num_data);
                for (int i = 0; i < num_data; i++)
                        dp_ids[i] = dataset[i].id;
                MPI_Scatterv(dp_ids.data(), sendcounts.data(), displs.data(),
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
                // All first row ranks tell master their recvcounts
                MPI_Gather(&dp_recvcount, 1, MPI_INT, nullptr, 1, nullptr, 0, col_comm);
                
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

        // print datapoints
        for (int r=0; r<size; r++) {
                MPI_Barrier(MPI_COMM_WORLD);
                if (rank == r) {
                        std::cout << "Datapoints received by rank " << rank << " from first row:\n";
                        for (int i = 0; i < dp_recvcount; i++) {
                                std::cout << "ID: " << dp_id_recv_buf[i] << ", Label: " << dp_label_recv_buf[i] << ", Attrs: ";
                                for (int a = 0; a < num_attrs; a++) {
                                        std::cout << dp_attr_recv_buf[i * num_attrs + a] << " ";
                                }
                                std::cout << "\n";
                        }
                }
        }

        // // ============================================================
        // // 3. Partition queries along first columns
        // // ============================================================
        // int q_remainder = num_queries % dims[1];
        // int q_recvcount = num_queries / dims[1] + (col == dims[1]-1 ? q_remainder : 0);

        // std::vector<Query> local_queries(q_recvcount);
        // std::vector<int> query_id_local(q_recvcount);
        // std::vector<int> query_k_local(q_recvcount);
        // std::vector<double> query_attr_local(q_recvcount * num_attrs);

        // // MASTER ONLY scatter queries to COLUMN ROOTS
        // if (rank == 0) {
        //         std::vector<int> q_sendcounts(dims[1]), q_displs(dims[1]);
        //         build_sendcounts_displs(num_queries, dims[1], col,
        //                         q_sendcounts, q_displs);
        //         std::vector<int> q_ids(num_queries), q_ks(num_queries);
        //         std::vector<double> q_attrs(num_queries * num_attrs);
        //         for (int i = 0; i < num_queries; i++) {
        //                 q_ids[i] = queries[i].id;
        //                 q_ks[i]  = queries[i].k;
        //                 for (int a = 0; a < num_attrs; a++)
        //                         q_attrs[i * num_attrs + a] = queries[i].attrs[a];
        //         }
        //         MPI_Scatterv(q_ids.data(), q_sendcounts.data(), q_displs.data(),
        //                         MPI_INT, query_id_local.data(), q_recvcount, MPI_INT,
        //                         0, row_comm);
        //         MPI_Scatterv(q_ks.data(), q_sendcounts.data(), q_displs.data(),
        //                         MPI_INT, query_k_local.data(), q_recvcount, MPI_INT,
        //                         0, row_comm);

        //         std::vector<int> q_attrs_sc(dims[1]), q_attrs_disp(dims[1]);
        //         build_sendcounts_displs_attrs(q_sendcounts, num_attrs,
        //                         q_attrs_sc, q_attrs_disp);
        //         std::vector<double> query_attr_local(q_recvcount * num_attrs);
        //         MPI_Scatterv(q_attrs.data(), q_attrs_sc.data(), q_attrs_disp.data(),
        //                         MPI_DOUBLE, query_attr_local.data(), q_recvcount*num_attrs,
        //                         MPI_DOUBLE, 0, row_comm);

        //         for (int i = 0; i < q_recvcount; i++) {
        //                 local_queries[i].id = query_id_local[i];
        //                 local_queries[i].k  = query_k_local[i];
        //                 local_queries[i].attrs.resize(num_attrs);
        //                 for (int a = 0; a < num_attrs; a++)
        //                         local_queries[i].attrs[a] = query_attr_local[i * num_attrs + a];
        //         }
        // }
        // // COLUMN ROOTS receive from MASTER
        // else if (col == 0) {
        //         MPI_Scatterv(nullptr, nullptr, nullptr,
        //                         MPI_INT, query_id_local.data(), q_recvcount, MPI_INT,
        //                         0, row_comm);

        //         MPI_Scatterv(nullptr, nullptr, nullptr,
        //                         MPI_INT, query_k_local.data(), q_recvcount, MPI_INT,
        //                         0, row_comm);

        //         std::vector<double> query_attr_local(q_recvcount * num_attrs);
        //         MPI_Scatterv(nullptr, nullptr, nullptr, MPI_DOUBLE,
        //                         query_attr_local.data(), q_recvcount*num_attrs,
        //                         MPI_DOUBLE, 0, row_comm);

        //         for (int i = 0; i < q_recvcount; i++) {
        //                 local_queries[i].id = query_id_local[i];
        //                 local_queries[i].k  = query_k_local[i];
        //                 local_queries[i].attrs.resize(num_attrs);
        //                 for (int a = 0; a < num_attrs; a++)
        //                         local_queries[i].attrs[a] = query_attr_local[i * num_attrs + a];
        //         }
        // }

        // // Broadcast queries down each row
        // for (int i = 0; i < q_recvcount; i++) {
        //         MPI_Bcast(&local_queries[i].id, 1, MPI_INT, 0, col_comm);
        //         MPI_Bcast(&local_queries[i].k,  1, MPI_INT, 0, col_comm);
        //         local_queries[i].attrs.resize(num_attrs);
        //         MPI_Bcast(local_queries[i].attrs.data(), num_attrs, MPI_DOUBLE, 0, col_comm);
        // }

        // // ============================================================
        // // 4. Define tuple type for (distance, label, id)
        // // ============================================================
        // struct tuple { double distance; int label; int id; };
        // tuple example;
        // MPI_Datatype tuple_type;
        // MPI_Datatype types[3] = {MPI_DOUBLE, MPI_INT, MPI_INT};
        // int blocklengths[3]  = {1,1,1};
        // MPI_Aint disp[3];

        // MPI_Get_address(&example.distance, &disp[0]);
        // MPI_Get_address(&example.label,    &disp[1]);
        // MPI_Get_address(&example.id,       &disp[2]);
        // MPI_Aint base = disp[0];
        // for (int i = 0; i < 3; i++) disp[i] -= base;

        // MPI_Type_create_struct(3, blocklengths, disp, types, &tuple_type);
        // MPI_Type_commit(&tuple_type);

        // // ============================================================
        // // 5. Local computation 
        // // ============================================================
        // std::vector<std::vector<tuple>> local_results(q_recvcount);

        // for (int qi = 0; qi < q_recvcount; qi++) {

        //         // for each query, compute distances to all local datapoints
        //         local_results[qi].clear();
        //         int k = local_queries.at(qi).k;
        //         for (int dp = 0; dp < dp_recvcount; dp++) {
        //                 double dist = computeDistance(
        //                                 local_queries.at(qi).attrs.data(),
        //                                 &dp_attr_recv_buf[dp * num_attrs],
        //                                 num_attrs
        //                                 );
        //                 local_results[qi].push_back({dist, dp_label_recv_buf.at(dp), dp_id_recv_buf.at(dp)});
        //         }

        //         // keep only top-k closest
        //         std::nth_element(local_results[qi].begin(),
        //                         local_results[qi].begin() + k,
        //                         local_results[qi].end(),
        //                         [](const tuple &a, const tuple &b){
        //                         if (a.distance == b.distance)
        //                         return a.label > b.label;
        //                         return a.distance < b.distance;
        //                         });
        //         local_results[qi].resize(k);
        // }

        // std::vector<tuple> local_results_flat;
        // std::vector<int> k_per_query(q_recvcount);
        // for (int qi = 0; qi < q_recvcount; qi++) {
        //         int k = local_results[qi].size(); 
        //         k_per_query[qi] = k;
        //         local_results_flat.insert(local_results_flat.end(),
        //                         local_results[qi].begin(),
        //                         local_results[qi].end());
        // }

        // // ============================================================
        // // 6. Gather local k-per-query counts to column roots
        // // ============================================================
        // std::vector<int> gathered_k_counts; // only valid on column root

        // // gather k_per_query from all processes in column
        // if (col == 0) {
        //         gathered_k_counts.resize(q_recvcount * dims[0]);
        // }
        // MPI_Gather(k_per_query.data(), q_recvcount, MPI_INT,
        //                 gathered_k_counts.data(), q_recvcount, MPI_INT,
        //                 0, col_comm);

        // // gather tuples using Gatherv with variable counts
        // std::vector<int> recvcounts(dims[0], 0);
        // std::vector<int> displs(dims[0], 0);
        // int sendcount = local_results_flat.size();

        // if (col == 0) {
        //         // compute recvcounts for each process based on gathered k_per_query
        //         for (int p = 0; p < dims[0]; p++) {
        //                 int sum_k = 0;
        //                 for (int qi = 0; qi < q_recvcount; qi++) {
        //                         sum_k += gathered_k_counts[p * q_recvcount + qi];
        //                 }
        //                 recvcounts[p] = sum_k;
        //         }
        //         displs[0] = 0;
        //         for (int p = 1; p < dims[0]; p++) {
        //                 displs[p] = displs[p - 1] + recvcounts[p - 1];
        //         }
        // }

        // // receive buffer on column root
        // std::vector<tuple> results_recv_buf;
        // if (col == 0) {
        //         int total_recv = 0;
        //         for (auto c : recvcounts) total_recv += c;
        //         results_recv_buf.resize(total_recv);
        // }

        // // gather tuples
        // MPI_Gatherv(local_results_flat.data(), sendcount, tuple_type,
        //                 col == 0 ? results_recv_buf.data() : nullptr,
        //                 col == 0 ? recvcounts.data() : nullptr,
        //                 col == 0 ? displs.data() : nullptr,
        //                 tuple_type, 0, col_comm);

        // // ============================================================
        // // 6b. Column root: assemble per-query gathered results
        // // ============================================================
        // std::vector<std::vector<tuple>> gathered_results;
        // if (col == 0) {
        //         gathered_results.resize(q_recvcount);
        //         int offset = 0;
        //         for (int p = 0; p < dims[0]; p++) {
        //                 for (int qi = 0; qi < q_recvcount; qi++) {
        //                         int k = gathered_k_counts[p * q_recvcount + qi];
        //                         gathered_results[qi].insert(
        //                                         gathered_results[qi].end(),
        //                                         results_recv_buf.begin() + offset,
        //                                         results_recv_buf.begin() + offset + k
        //                                         );
        //                         offset += k;
        //                 }
        //         }

        //         // keep only top-k per query
        //         for (int qi = 0; qi < q_recvcount; qi++) {
        //                 int k = local_queries[qi].k;
        //                 auto &vec = gathered_results[qi];
        //                 std::nth_element(vec.begin(), vec.begin() + k, vec.end(),
        //                                 [](const tuple &a, const tuple &b){
        //                                 if (a.distance == b.distance)
        //                                 return a.label > b.label;
        //                                 return a.distance < b.distance;
        //                                 });
        //                 vec.resize(k);
        //         }
        // }

        // // ============================================================
        // // 7. Gather final results from column roots to MASTER (rank 0)
        // // ============================================================
        // std::vector<int> column_recvcounts(dims[1], 0);
        // std::vector<int> column_displs(dims[1], 0);
        // std::vector<tuple> send_buf;

        // if (col == 0) {
        //         // flatten gathered_results
        //         for (int qi = 0; qi < q_recvcount; qi++) {
        //                 send_buf.insert(send_buf.end(),
        //                                 gathered_results[qi].begin(),
        //                                 gathered_results[qi].end());
        //         }
        //         if (row == 1) {
        //             // print contents
        //             std::cout << "Row 1 sending " << send_buf.size() << " tuples to master." << std::endl;
        //             for (auto &tup : send_buf) {
        //                 std::cout << "Tuple - Distance: " << tup.distance << ", Label: " << tup.label << ", ID: " << tup.id << std::endl;
        //             }
        //         }
        // }

        // int sendcount_final = send_buf.size();
        // int recvcount_final = 0;
        // MPI_Reduce(&sendcount_final, &recvcount_final, 1, MPI_INT, MPI_SUM, 0, row_comm);

        // // receive buffer on MASTER
        // std::vector<tuple> final_results_recv_buf;
        // if (rank == 0) {
        //         final_results_recv_buf.resize(recvcount_final);
        // }
        // MPI_Gather(&sendcount_final, 1, MPI_INT,
        //                 column_recvcounts.data(), 1, MPI_INT,
        //                 0, row_comm);

        // // prepare displacements for MASTER
        // if (rank == 0) {
        //         column_displs[0] = 0;
        //         for (int i = 1; i < dims[1]; i++) {
        //                 column_displs[i] = column_displs[i-1] + column_recvcounts[i-1];
        //         }
        // }

        //TODO: ids are messed up in send_buf, fix gather here

        // gather tuples
        // MPI_Gatherv(col == 0 ? send_buf.data() : nullptr,
        //     col == 0 ? send_buf.size() : 0,
        //     tuple_type,
        //     rank == 0 ? final_results_recv_buf.data() : nullptr,
        //     rank == 0 ? column_recvcounts.data() : nullptr,
        //     rank == 0 ? column_displs.data() : nullptr,
        //     tuple_type, 0, row_comm);

        // ============================================================
        // 8. MASTER reconstructs per-query results and predicts
        // ============================================================
        // if (rank == 0) {
        //         std::unordered_map<int, std::vector<tuple>> results_map;
        //         int offset = 0;
        //         for (int c = 0; c < dims[1]; c++) {
        //                 for (int qi = 0; qi < q_recvcount; qi++) {
        //                         int k = gathered_results[qi].size(); // number of tuples per query per column
        //                         results_map[qi].insert(results_map[qi].end(),
        //                                         final_results_recv_buf.begin() + offset,
        //                                         final_results_recv_buf.begin() + offset + k);
        //                         offset += k;
        //                 }
        //         }

        //         // predict and report
        //         for (int qi = 0; qi < num_queries; qi++) {
        //                 auto &res = results_map[qi];
        //                 // vote
        //                 std::unordered_map<int, int> label_count;
        //                 for (auto &tup : res) label_count[tup.label]++;
        //                 int max_count = 0, predicted_label = -1;
        //                 for (auto &lc : label_count) {
        //                         if (lc.second > max_count || (lc.second == max_count && lc.first > predicted_label)) {
        //                                 max_count = lc.second;
        //                                 predicted_label = lc.first;
        //                         }
        //                 }
        //                 // sort by distance then ID
        //                 std::sort(res.begin(), res.end(),
        //                                 [](const tuple &a, const tuple &b){
        //                                 if (a.distance == b.distance) return a.id > b.id;
        //                                 return a.distance < b.distance;
        //                                 });
        //                 // convert to (distance, label)
        //                 std::vector<std::pair<double,int>> res_pairs;
        //                 for (auto &tup : res) res_pairs.emplace_back(tup.distance, tup.label);
        //                 reportResult(queries[qi], res_pairs, predicted_label);
        //         }
        // }
}
