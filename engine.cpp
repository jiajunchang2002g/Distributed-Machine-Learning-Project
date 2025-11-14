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
    int Pd = floor(sqrt(size));
    while (Pd > 1 && size % Pd != 0) Pd--;  // factor
    int Pq = size / Pd;

    if (rank == 0)
        std::cout << "Using 2D grid Pd=" << Pd << " Pq=" << Pq << "\n";

    int dims[2] = {Pd, Pq};
    int periods[2] = {0, 0};
    MPI_Comm cart;
    MPI_Cart_create(world, 2, dims, periods, 0, &cart);

    int coords[2];
    MPI_Cart_coords(cart, rank, 2, coords);
    int row = coords[0];
    int col = coords[1];

    // Create row/column communicators
    int row_dims[2] = {1, 0};  // keep row fixed
    int col_dims[2] = {0, 1};  // keep col fixed

    MPI_Comm row_comm, col_comm;
    MPI_Cart_sub(cart, row_dims, &row_comm);
    MPI_Cart_sub(cart, col_dims, &col_comm);

    // print grid info
    std::cout << "Rank " << rank << " at (" << row << "," << col << ")\n";  

    // print row/col communicators
    int row_rank, col_rank;
    MPI_Comm_rank(row_comm, &row_rank);
    MPI_Comm_rank(col_comm, &col_rank);
    std::cout << "Rank " << rank << " has row_comm rank " << row_rank 
              << " and col_comm rank " << col_rank << "\n";

    // ============================================================
    // 2. Partition datapoints along Pd rows
    // ============================================================
    int dp_per_row = num_data / Pd;
    int dp_rem     = num_data % Pd;
    int dp_local   = dp_per_row + (row == Pd-1 ? dp_rem : 0);

    std::vector<int>    dp_id_local(dp_local);
    std::vector<int>    dp_label_local(dp_local);
    std::vector<double> dp_attr_local(dp_local * num_attrs);

    if (rank == 0) {
        // Build sendcounts/displs for rows
        std::vector<int> r_sendcounts(Pd), r_displs(Pd);
        build_sendcounts_displs(num_data, Pd, row, r_sendcounts, r_displs);

        // Scatter ids + labels
        MPI_Scatterv(dataset.data(), r_sendcounts.data(), r_displs.data(),
                     MPI_INT, dp_id_local.data(), dp_local, MPI_INT,
                     0, row_comm);

        std::vector<int> labels(num_data);
        for (int i = 0; i < num_data; i++)
            labels[i] = dataset[i].label;

        MPI_Scatterv(labels.data(), r_sendcounts.data(), r_displs.data(),
                     MPI_INT, dp_label_local.data(), dp_local, MPI_INT,
                     0, row_comm);

        // Scatter attributes
        std::vector<double> all_attrs(num_data * num_attrs);
        for (int i = 0; i < num_data; i++)
            for (int a = 0; a < num_attrs; a++)
                all_attrs[i * num_attrs + a] = dataset[i].attrs[a];

        // compute attr_sendcounts/displs
        std::vector<int> attr_sc(Pd), attr_disp(Pd);
        build_sendcounts_displs_attrs(r_sendcounts, num_attrs,
                                         attr_sc, attr_disp);

        MPI_Scatterv(all_attrs.data(), attr_sc.data(), attr_disp.data(),
                     MPI_DOUBLE, dp_attr_local.data(), dp_local*num_attrs,
                     MPI_DOUBLE, 0, row_comm);
    }
    else if (col == 0) {
        // row_comm receivers
        MPI_Scatterv(nullptr, nullptr, nullptr,
                     MPI_INT, dp_id_local.data(), dp_local, MPI_INT,
                     0, row_comm);

        MPI_Scatterv(nullptr, nullptr, nullptr,
                     MPI_INT, dp_label_local.data(), dp_local, MPI_INT,
                     0, row_comm);

        MPI_Scatterv(nullptr, nullptr, nullptr,
                     MPI_DOUBLE, dp_attr_local.data(), dp_local*num_attrs,
                     MPI_DOUBLE, 0, row_comm);
    }

    // Broadcast datapoints from row leaders to entire row
    MPI_Bcast(dp_id_local.data(),    dp_local,         MPI_INT,    0, row_comm);
    MPI_Bcast(dp_label_local.data(), dp_local,         MPI_INT,    0, row_comm);
    MPI_Bcast(dp_attr_local.data(),  dp_local*num_attrs, MPI_DOUBLE, 0, row_comm);

    // ============================================================
    // 3. Partition queries along Pq columns
    // ============================================================
    int q_per_col = num_queries / Pq;
    int q_rem     = num_queries % Pq;
    int q_local   = q_per_col + (col == Pq-1 ? q_rem : 0);

    std::vector<Query> local_queries(q_local);

    std::vector<int> query_id_local(q_local);
    std::vector<int> query_k_local(q_local);
    std::vector<double> query_attr_local(q_local * num_attrs);

    if (rank == 0) {
        // Build query sendcounts for columns
        std::vector<int> q_sendcounts(Pq), q_displs(Pq);
        for (int c = 0; c < Pq; c++)
            q_sendcounts[c] = q_per_col + (c == Pq-1 ? q_rem : 0);

        q_displs[0] = 0;
        for (int c = 1; c < Pq; c++)
            q_displs[c] = q_displs[c-1] + q_sendcounts[c-1];

        // convert queries to separate arrays
        std::vector<int> q_ids(num_queries), q_ks(num_queries);
        std::vector<double> q_attrs(num_queries * num_attrs);

        for (int i = 0; i < num_queries; i++) {
            q_ids[i] = queries[i].id;
            q_ks[i]  = queries[i].k;
            for (int a = 0; a < num_attrs; a++)
                q_attrs[i * num_attrs + a] = queries[i].attrs[a];
        }

        // Scatter ids
        MPI_Scatterv(q_ids.data(), q_sendcounts.data(), q_displs.data(),
                     MPI_INT, query_id_local.data(), q_local, MPI_INT,
                     0, col_comm);

        // Scatter ks (each column leader gets full chunk)
        MPI_Scatterv(q_ks.data(), q_sendcounts.data(), q_displs.data(),
                     MPI_INT, query_k_local.data(), q_local, MPI_INT,
                     0, col_comm);

        // Scatter attrs
        std::vector<int> qa_sc(Pq), qa_disp(Pq);
        for (int c = 0; c < Pq; c++)
            qa_sc[c] = q_sendcounts[c] * num_attrs;
        qa_disp[0] = 0;
        for (int c = 1; c < Pq; c++)
            qa_disp[c] = qa_disp[c-1] + qa_sc[c-1];

        std::vector<double> query_attr_local(q_local * num_attrs);
        MPI_Scatterv(q_attrs.data(), qa_sc.data(), qa_disp.data(),
                     MPI_DOUBLE, query_attr_local.data(), q_local*num_attrs,
                     MPI_DOUBLE, 0, col_comm);

        // Fill local query vector
        for (int i = 0; i < q_local; i++) {
            local_queries[i].id = query_id_local[i];
            local_queries[i].k  = query_k_local[i];
            local_queries[i].attrs.resize(num_attrs);
            for (int a = 0; a < num_attrs; a++)
                local_queries[i].attrs[a] = query_attr_local[i * num_attrs + a];
        }
    }
    else if (row == 0) {
        // column leaders
        MPI_Scatterv(nullptr, nullptr, nullptr,
                     MPI_INT, query_id_local.data(), q_local, MPI_INT,
                     0, col_comm);

        MPI_Scatterv(nullptr, nullptr, nullptr,
                     MPI_INT, query_k_local.data(), q_local, MPI_INT,
                     0, col_comm);

        std::vector<double> query_attr_local(q_local * num_attrs);
        MPI_Scatterv(nullptr, nullptr, nullptr, MPI_DOUBLE,
                     query_attr_local.data(), q_local*num_attrs,
                     MPI_DOUBLE, 0, col_comm);

        for (int i = 0; i < q_local; i++) {
            local_queries[i].id = query_id_local[i];
            local_queries[i].k  = query_k_local[i];
            local_queries[i].attrs.resize(num_attrs);
            for (int a = 0; a < num_attrs; a++)
                local_queries[i].attrs[a] = query_attr_local[i * num_attrs + a];
        }
    }

    // Broadcast queries down each column
    for (int i = 0; i < q_local; i++) {
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
    // 5. Local computation: each rank handles q_local queries
    // ============================================================
    std::vector<tuple> local_knn;
    local_knn.reserve(dp_local);

    // For reduce result on column roots
    std::vector<tuple> col_knn_local;
    // each query will have its own k value 
    // std::vector<tuple> col_knn_final; 

    // ============================================================
    // 6. Process each local query
    // ============================================================
    for (int qi = 0; qi < q_local; qi++) {

        local_knn.clear();
        int k = local_queries.at(qi).k;

        // Compute local distances
        for (int dp = 0; dp < dp_local; dp++) {
            double dist = computeDistance(
                local_queries.at(qi).attrs.data(),
                &dp_attr_local[dp * num_attrs],
                num_attrs
            );
            local_knn.push_back({dist, dp_label_local.at(dp), dp_id_local.at(dp)});
        }

        // Keep local top k
        std::nth_element(local_knn.begin(),
                         local_knn.begin() + k,
                         local_knn.end(),
                         [](const tuple &a, const tuple &b){
                             if (a.distance == b.distance)
                                 return a.label > b.label;
                             return a.distance < b.distance;
                         });

        // resize to k
        local_knn.resize(k);

        // // =====================================================
        // // Reduce along *datapoint* dimension (col_comm)
        // // =====================================================
        // col_knn_local = local_knn;  // local contributor

        // MPI_Reduce(col_knn_local.data(), col_knn_final.data(),
        //            k, tuple_type,        // count = k
        //            // custom op not needed: we reduce by performing a Gather then sort
        //            MPI_MAX,              // placeholder (we gather differently)
        //            0, col_comm);

        // // NOTE:
        // // True reduction requires custom operator. To keep code simpler,
        // // we gather then merge on column root:
        // // (This section below implements merging)
        // if (col == 0) {
        //     // Each rank sends top-k, gather them:
        //     std::vector<tuple> gathered(Pd * k);

        //     MPI_Gather(col_knn_local.data(), k, tuple_type,
        //                gathered.data(),      k, tuple_type,
        //                0, col_comm);

        //     // merge & sort
        //     std::sort(gathered.begin(), gathered.end(),
        //               [](const tuple &a, const tuple &b){
        //                   if (a.distance == b.distance)
        //                       return a.label > b.label;
        //                   return a.distance < b.distance;
        //               });

        //     // Keep final top k
        //     for (int i = 0; i < k; i++)
        //         col_knn_final[i] = gathered[i];
        // }

        // =====================================================
        // Column root now has final kNN for this query
        // =====================================================
    }

    // // ============================================================
    // // 7. Gather final results from column roots to rank 0
    // // ============================================================
    // if (col == 0) {
    //     // send q_local blocks to rank 0
    //     MPI_Gather(col_knn_final.data(), q_local * queries[0].k, tuple_type,
    //                (rank == 0 ? MPI_IN_PLACE : nullptr),
    //                q_local * queries[0].k, tuple_type,
    //                0, world);
    // }
    // else {
    //     MPI_Gather(nullptr, 0, tuple_type,
    //                nullptr, 0, tuple_type,
    //                0, world);
    // }

    // // ============================================================
    // // 8. Rank 0 reconstructs full results and prints
    // // ============================================================
    // if (rank == 0) {

    //     // reconstruct and report query results
    //     // (full assembly omitted for brevity; follows your original code)

    //     std::cout << "All queries processed in 2D decomposition.\n";
    //     // call reportResult() in order for each query
    // }

    MPI_Type_free(&tuple_type);
}

