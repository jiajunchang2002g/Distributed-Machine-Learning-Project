#include <vector>
#include <iostream>
#include "engine.h"

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
        int sendcount, recvcount;
        MPI_Comm comm = MPI_COMM_WORLD;
        MPI_Status Stat;
        MPI_Comm_size(comm, &numtasks);
        MPI_Comm_rank(comm, &rank);

        sendcount = p.num_data / numtasks;
        recvcount = p.num_data / numtasks;

        // recvbuffer(s)
        double **attrs_rx = (double **)malloc(sizeof(double *) * recvcount);
        for (int i = 0; i < recvcount; i++)
        {
                attrs_rx[i] = (double *)malloc(sizeof(double) * p.num_attrs);
        }
        int *id_rx = (int *)malloc(sizeof(int) * recvcount);
        int *label_rx = (int *)malloc(sizeof(int) * recvcount);
        double **q_attrs_rx = (double **)malloc(sizeof(double *) * p.num_attrs);

        // sendbuffers(s)
        double **attrs_tx;
        int *id_tx;
        int *label_tx;
        if (rank == 0)
        {
                attrs_tx = (double **)malloc(sizeof(double *) * sendcount);
                for (int i = 0; i < sendcount; i++)
                {
                        attrs_tx[i] = (double *)malloc(sizeof(double) * p.num_attrs);
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

        for (int i = 0; i < p.num_queries; i++)
        {
                int query_id;
                int query_k;
                double *query_attrs = (double *)malloc(sizeof(double) * p.num_attrs);
                if (rank == 0)
                {
                        query_id = queries[i].id;
                        query_k = queries[i].k;
                        query_attrs = queries[i].attrs.data();
                }
                MPI_Bcast(&query_id, 1, MPI_INT, 0, comm);
                MPI_Bcast(&query_k, 1, MPI_INT, 0, comm);
                MPI_Bcast(query_attrs, p.num_attrs, MPI_DOUBLE, 0, comm);

                MPI_Scatter(id_tx, sendcount, MPI_INT, id_rx, recvcount, MPI_INT, 0, comm);
                MPI_Scatter(label_tx, sendcount, MPI_INT, label_rx, recvcount, MPI_INT, 0, comm);
                MPI_Scatter(attrs_tx[0], sendcount * p.num_attrs, MPI_DOUBLE,
                            attrs_rx[0], recvcount * p.num_attrs, MPI_DOUBLE, 0, comm);
                
                
        }
        printf("rank= %d \n", rank);
}
