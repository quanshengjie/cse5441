#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int niters;
int nelems;
int rank;
int size;
double *A;
double *B;

void setup(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc!=3 || size!=2) {
        printf("Usage: mpiexec -n 2 ./%s <nbytes> <niters>\n", argv[0]);
    }

    nelems = atoi(argv[1]);
    niters = atoi(argv[2]);
    A = malloc(nelems * sizeof (double));
    B = malloc(nelems * sizeof (double));
}

void pingpong_bw()
{
    int i;
    double bw;
    double t1, t2;
    MPI_Request requests[4];

    if (rank==0) {
        MPI_Barrier(MPI_COMM_WORLD);
        t1 = MPI_Wtime();
        for (i=0; i<niters; i++) {
            MPI_Isend(A, nelems, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, &requests[0]);
            MPI_Irecv(B, nelems, MPI_DOUBLE, 1, 2, MPI_COMM_WORLD, &requests[1]);
            MPI_Isend(B, nelems, MPI_DOUBLE, 1, 3, MPI_COMM_WORLD, &requests[2]);
            MPI_Irecv(A, nelems, MPI_DOUBLE, 1, 4, MPI_COMM_WORLD, &requests[3]);
            MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
        }
        t2 = MPI_Wtime();
        bw = 8*nelems/((t2-t1)/(4*niters));
        printf("size: %7d\tbandwidth: %lf MBps\n", nelems, bw/1.0e6);
    } else {
        MPI_Barrier(MPI_COMM_WORLD);
        for (i=0; i<niters; i++) {
            MPI_Irecv(A, nelems, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &requests[0]);
            MPI_Isend(A, nelems, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, &requests[1]);
            MPI_Irecv(B, nelems, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD, &requests[2]);
            MPI_Isend(B, nelems, MPI_DOUBLE, 0, 4, MPI_COMM_WORLD, &requests[3]);
            MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
        }
    }
}

void cleanup()
{
    free(A);
    free(B);

    MPI_Finalize();
}

int main(int argc, char **argv)
{
    setup(argc, argv);

    pingpong_bw();

    cleanup();
    return 0;
}
