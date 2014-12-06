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
    if (argc!=3) {
        printf("Usage: ./%s <nbytes> <niters>\n", argv[0]);
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

    if (rank==0) {
        MPI_Barrier(MPI_COMM_WORLD);
        t1 = MPI_Wtime();
        for (i=0; i<niters; i++) {
            MPI_Send(A, nelems, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD);
            MPI_Recv(B, nelems, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(B, nelems, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD);
            MPI_Recv(A, nelems, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        t2 = MPI_Wtime();
        bw = 8*nelems/((t2-t1)/(4*niters));
        printf("bandwidth = %lf MBps\n", bw/1.0e6);
    } else {
        MPI_Barrier(MPI_COMM_WORLD);
        for (i=0; i<niters; i++) {
            MPI_Recv(A, nelems, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(A, nelems, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
            MPI_Recv(B, nelems, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(B, nelems, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
        }
    }
}

void cleanup()
{
    free(A);
    free(B);
}

int main(int argc, char **argv)
{
    setup(argc, argv);

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    pingpong_bw();

    MPI_Finalize();
    return 0;
}
