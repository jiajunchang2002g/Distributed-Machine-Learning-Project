#include "engine.h"
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
        int row_dims[2] = {false, true};  // fix row 
        int col_dims[2] = {true, false};  // fix column

        MPI_Comm row_comm, col_comm;
        MPI_Cart_sub(cart, row_dims, &row_comm);        // ranks have same row: same datapoints, different queries
        MPI_Cart_sub(cart, col_dims, &col_comm);        // ranks have same column: same queries, different datapoints

        // ============================================================
        // 2. Scatter datapoints from master to first column of workers
        // ============================================================
        int dp_remainder = num_data % dims[0];
        int dp_recvcount = num_data / dims[0] + (row == 0 ? dp_remainder : 0);  // root and its row handle remainder

        std::vector<int>    dp_id_recv_buf(dp_recvcount);
        std::vector<int>    dp_label_recv_buf(dp_recvcount);
        std::vector<double> dp_attr_recv_buf(dp_recvcount * num_attrs);         // index == dpi * num_attrs

        // MASTER ONLY
        if (rank == 0) {
                std::vector<int> dp_sendcounts(dims[0]), dp_displs(dims[0]);

                MPI_Gather(&dp_recvcount, 1, MPI_INT, dp_sendcounts.data(), 1, MPI_INT, 0, col_comm);
                dp_displs[0] = 0;
                for (int i = 1; i < dims[0]; i++)
                        dp_displs[i] = dp_displs[i-1] + dp_sendcounts[i-1];

                // Scatter datapoint IDs
                std::vector<int> dp_ids(num_data);
                for (int i = 0; i < num_data; i++)
                        dp_ids[i] = dataset[i].id;
                MPI_Scatterv(dp_ids.data(), dp_sendcounts.data(), dp_displs.data(),
                                MPI_INT, dp_id_recv_buf.data(), dp_recvcount, MPI_INT,
                                0, col_comm);
                // Scatter datapoint labels
                std::vector<int> labels(num_data);
                for (int i = 0; i < num_data; i++)
                        labels[i] = dataset[i].label;
                MPI_Scatterv(labels.data(), dp_sendcounts.data(), dp_displs.data(),
                                MPI_INT, dp_label_recv_buf.data(), dp_recvcount, MPI_INT,
                                0, col_comm);
                // Scatter datapoint attributes
                std::vector<double> all_attrs(num_data * num_attrs);
                for (int i = 0; i < num_data; i++)
                        for (int a = 0; a < num_attrs; a++)
                                all_attrs[i * num_attrs + a] = dataset[i].attrs[a];
                std::vector<int> attr_sendcounts(dims[0]), attr_displs(dims[0]);
                for (int i = 0; i < dims[0]; i++) {
                        attr_sendcounts[i]   = dp_sendcounts[i] * num_attrs;
                }
                attr_displs[0] = 0;
                for (int i = 1; i < dims[0]; i++) {
                        attr_displs[i] = attr_displs[i-1] + attr_sendcounts[i-1];
                }
                
                MPI_Scatterv(all_attrs.data(), attr_sendcounts.data(), attr_displs.data(),
                                MPI_DOUBLE, dp_attr_recv_buf.data(), dp_recvcount*num_attrs,
                                MPI_DOUBLE, 0, col_comm);
        } 

        // FIRST COL receive from MASTER
        else if (col == 0) {
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

        // // FIRST COL broadcast to OTHER COLS
        MPI_Bcast(dp_id_recv_buf.data(), dp_recvcount, MPI_INT, 0, row_comm);
        MPI_Bcast(dp_label_recv_buf.data(), dp_recvcount, MPI_INT, 0, row_comm);
        MPI_Bcast(dp_attr_recv_buf.data(), dp_recvcount * num_attrs, MPI_DOUBLE, 0, row_comm);

        // ============================================================
        // 3. Scatter queries from master to first row of workers
        // ============================================================
        int q_remainder = num_queries % dims[1];
        int q_recvcount = num_queries / dims[1] + (col == 0 ? q_remainder : 0);

        std::vector<int> query_id_recv_buf(q_recvcount);
        std::vector<int> query_k_recv_buf(q_recvcount);
        std::vector<double> query_attr_recv_buf(q_recvcount * num_attrs);       // index == qi * num_attrs

        // MASTER ONLY scatter queries to first row 
        if (rank == 0) {
                std::vector<int> q_sendcounts(dims[1]), q_displs(dims[1]);

                MPI_Gather(&q_recvcount, 1, MPI_INT, q_sendcounts.data(), 1, MPI_INT, 0, row_comm);

                q_displs[0] = 0;
                for (int i = 1; i < dims[1]; i++)
                        q_displs[i] = q_displs[i-1] + q_sendcounts[i-1];

                // Scatter query id 
                std::vector<int> query_id_send_buf(num_queries);
                for (int i = 0; i < num_queries; i++) {
                        query_id_send_buf[i] = queries[i].id;
                }
                MPI_Scatterv(query_id_send_buf.data(), q_sendcounts.data(), q_displs.data(),
                                MPI_INT, query_id_recv_buf.data(), q_recvcount, MPI_INT,
                                0, row_comm);
                // Scatter query k
                std::vector<int> query_k_send_buf(num_queries);
                for (int i = 0; i < num_queries; i++) {
                        query_k_send_buf[i] = queries[i].k;
                }
                MPI_Scatterv(query_k_send_buf.data(), q_sendcounts.data(), q_displs.data(),
                                MPI_INT, query_k_recv_buf.data(), q_recvcount, MPI_INT,
                                0, row_comm);

                std::vector<int> q_attrs_sendcounts(dims[1]), q_attrs_disp(dims[1]);
                for (int i = 0; i < dims[1]; i++) {
                        q_attrs_sendcounts[i]   = q_sendcounts[i] * num_attrs;
                }
                q_attrs_disp[0] = 0;
                for (int i = 1; i < dims[1]; i++) {
                        q_attrs_disp[i] = q_attrs_disp[i-1] + q_attrs_sendcounts[i-1];
                }

                // Scatter query attrs
                std::vector<double> q_attrs(num_queries * num_attrs);
                for (int i = 0; i < num_queries; i++) {
                        for (int a = 0; a < num_attrs; a++)
                                q_attrs[i * num_attrs + a] = queries[i].attrs[a];
                }
                MPI_Scatterv(q_attrs.data(), q_attrs_sendcounts.data(), q_attrs_disp.data(),
                                MPI_DOUBLE, query_attr_recv_buf.data(), q_recvcount*num_attrs,
                                MPI_DOUBLE, 0, row_comm);
        }
        // First row ONLY receive 
        else if (row == 0) {
                MPI_Gather(&q_recvcount, 1, MPI_INT, nullptr, 1, nullptr, 0, row_comm);

                MPI_Scatterv(nullptr, nullptr, nullptr,
                                MPI_INT, query_id_recv_buf.data(), q_recvcount, MPI_INT,
                                0, row_comm);

                MPI_Scatterv(nullptr, nullptr, nullptr,
                                MPI_INT, query_k_recv_buf.data(), q_recvcount, MPI_INT,
                                0, row_comm);

                MPI_Scatterv(nullptr, nullptr, nullptr, MPI_DOUBLE,
                                query_attr_recv_buf.data(), q_recvcount*num_attrs,
                                MPI_DOUBLE, 0, row_comm);
        }

        // Broadcast queries from first row to OTHER ROWS
        MPI_Bcast(query_id_recv_buf.data(),    q_recvcount,         MPI_INT,    0, col_comm);
        MPI_Bcast(query_k_recv_buf.data(),     q_recvcount,         MPI_INT,    0, col_comm);
        MPI_Bcast(query_attr_recv_buf.data(),  q_recvcount*num_attrs, MPI_DOUBLE, 0, col_comm);

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

                std::vector<tuple> &local_result = local_results[qi];
                int query_k = query_k_recv_buf.at(qi);
                for (int dp = 0; dp < dp_recvcount; dp++) {
                        double dist = computeDistance(
                                        &query_attr_recv_buf[qi * num_attrs],
                                        &dp_attr_recv_buf[dp * num_attrs],
                                        num_attrs
                                        );
                        local_result.push_back({dist, dp_label_recv_buf.at(dp), dp_id_recv_buf.at(dp)});
                }

                // keep only top-k closest
                std::nth_element(local_result.begin(), local_result.begin() + query_k,
                             local_result.end(), [](const tuple &a, const tuple &b) {
                                   if (a.distance == b.distance) {
                                         return a.label > b.label; // larger label first
                                   }
                                   return a.distance < b.distance; // smaller distance first
                             });
                local_result.resize(query_k);
        }

        std::vector<tuple> local_results_flat;                          // index == sum_{j=0}^{qi-1} k_j + ki
        std::vector<int> query_k(q_recvcount);
        for (int qi = 0; qi < q_recvcount; qi++) {
                query_k[qi] = local_results[qi].size(); 
                local_results_flat.insert(
                        local_results_flat.end(),
                        local_results[qi].begin(),
                        local_results[qi].end()
                );
        }

        // ============================================================
        // 6. Gather all flat local results to first row
        // ============================================================
        int res_sendcount = local_results_flat.size(); // single send, unique to each col
        int res_recvcount = local_results_flat.size(); // single recv, unique to each col
        int res_recv_buf_size = local_results_flat.size() * dims[1]; // dims[1] == num cols
        std::vector<tuple> res_recvbuffer;

        // Gather results among ranks with same query shard at first row
        if (row == 0) {
                res_recvbuffer.resize(res_recv_buf_size);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Gather(local_results_flat.data(), local_results_flat.size(), 
                                tuple_type, res_recvbuffer.data(), res_recvcount, tuple_type, 0, col_comm); 

        // ============================================================
        // 6b. First row : assemble per-query gathered results
        // ============================================================
        std::vector<std::vector<tuple>> gathered_results;
        if (row == 0) {
                gathered_results.resize(q_recvcount);
                for (int qi = 0; qi < q_recvcount; qi++) {
                        std::vector<tuple> &gathered_result = gathered_results[qi];
                        for (int c = 0; c < dims[1]; c++) {
                                int offset = c * res_recvcount + qi * query_k[qi];
                                for (int ki = 0; ki < query_k[qi]; ki++) {
                                        gathered_result.push_back(res_recvbuffer[offset + ki]);
                                }
                        }
                        std::nth_element(gathered_result.begin(), gathered_result.begin() + query_k[qi],
                                     gathered_result.end(), [](const tuple &a, const tuple &b) {
                                           if (a.distance == b.distance) {
                                                 return a.label > b.label; // larger label first
                                           }
                                           return a.distance < b.distance; // smaller distance first
                                     });
                        gathered_result.resize(query_k[qi]);
                }
        }

        // ============================================================
        // 7. First row process results and report one by one
        // ============================================================
        for (int r=0; r<size; r++) {
                MPI_Barrier(MPI_COMM_WORLD);
                if (row == 0) {
                        for (int qi = 0; qi < q_recvcount; qi++) {
                                auto &res = gathered_results[qi];
                                // vote
                                std::unordered_map<int, int> label_count;
                                for (auto &tup : res) label_count[tup.label]++;
                                int max_count = 0, predicted_label = -1;
                                for (auto &lc : label_count) {
                                        if (lc.second > max_count || (lc.second == max_count && lc.first > predicted_label)) {
                                                max_count = lc.second;
                                                predicted_label = lc.first;
                                        }
                                }
                                // sort by distance then ID
                                std::sort(res.begin(), res.end(),
                                                [](const tuple &a, const tuple &b){
                                                if (a.distance == b.distance) return a.id > b.id;
                                                return a.distance < b.distance;
                                                });
                                // convert to (distance, label)
                                std::vector<std::pair<double,int>> res_pairs;
                                for (auto &tup : res) res_pairs.emplace_back(tup.distance, tup.label);
                                // make query
                                Query query = {
                                        query_id_recv_buf[qi],
                                        query_k_recv_buf[qi],
                                        std::vector<double>( &query_attr_recv_buf[qi * num_attrs],
                                                        &query_attr_recv_buf[(qi+1) * num_attrs] )
                                };
                                reportResult(query, res_pairs, predicted_label);
                        }
                }
        }
}

