#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>

#define N 1024
#define diag 5.0
#define recipdiag 0.2
#define odiag -1.0
#define eps  1.0E-6
#define maxiter 10000
#define nprfreq 10

int iter=-1;
int old=0, new=1, tmp;
double clkbegin, clkend;
double t;
int rank;
int size;
int ibegin, iend, chunk;
double rtclock();

void init(double**,double*);
double rhocalc(double*);
void update(double**,double*,double*); 
//     N is size of physical grid from which the sparse system is derived
//     diag is the value of each diagonal element of the sparse matrix
//     recipdiag is the reciprocal of each diagonal element 
//     odiag is the value of each off-diagonal non-zero element in matrix
//     eps is the threshold for the convergence criterion
//     nprfreq is the number of iterations per output of currwnt residual
//     xnew and xold are used to hold the N*N guess solution vector components
//     resid holds the current residual
//     rhoinit, rhonew hold the initial and current residual error norms, resp.
int main (int argc, char * argv[])
{
    int i,j;
    int row1, row2, row3;
    int rank1, rank2, rank3;
    double rhoinit,rhonew; 
    double *b, *x[2], *resid;
    double s1[2], s2[1], s3[2];
    MPI_Request requests[6];

    b = malloc(sizeof(double)*((N+2)*(N+2)));
    x[old] = malloc(sizeof(double)*((N+2)*(N+2)));
    x[new] = malloc(sizeof(double)*((N+2)*(N+2)));
    resid = malloc(sizeof(double)*((N+2)*(N+2)));

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    chunk = N/size;
    ibegin = (rank*chunk)+1;
    iend = ibegin + chunk;

    row1 = (N+2)/4;
    row2 = (N+1)/2;
    row3 = 3*(N+2)/4;
    rank1 = row1/chunk;
    rank2 = row2/chunk;
    rank3 = row3/chunk;

    init(x,b);
    rhoinit = rhocalc(b);

    MPI_Barrier(MPI_COMM_WORLD);
    clkbegin = rtclock();

    for(iter=0;iter<maxiter;iter++){
        update(x,resid,b);
        rhonew = rhocalc(resid);

        if(rhonew<eps){
            if(rank==rank1) {
                s1[0] = x[new][row1*(N+2)+row1];
                s1[1] = x[new][row1*(N+2)+row3];
                MPI_Isend(s1, 2, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &requests[0]);
            }
            if(rank==rank2) {
                s2[0] = x[new][row2*(N+2)+row2];
                MPI_Isend(s2, 1, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, &requests[1]);
            }
            if(rank==rank3) {
                s3[0] = x[new][row3*(N+2)+row1];
                s3[1] = x[new][row3*(N+2)+row3];
                MPI_Isend(s3, 2, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD, &requests[2]);
            }

            if(rank == 0) {
                MPI_Irecv(s1, 2, MPI_DOUBLE, rank1, 1, MPI_COMM_WORLD, &requests[3]);
                MPI_Irecv(s2, 1, MPI_DOUBLE, rank2, 2, MPI_COMM_WORLD, &requests[4]);
                MPI_Irecv(s3, 2, MPI_DOUBLE, rank3, 3, MPI_COMM_WORLD, &requests[5]);
                MPI_Waitall(3, &requests[3], MPI_STATUSES_IGNORE);

                clkend = rtclock();
                t = clkend-clkbegin;
                printf("Solution converged in %d iterations\n",iter);
                printf("Final residual norm = %f\n",rhonew);
                printf("Solution at center and four corners of interior N/2 by N/2 grid : \n");
                i=row1; j=row1; printf("xnew[%d][%d]=%f\n",i,j,s1[0]);
                i=row1; j=row3; printf("xnew[%d][%d]=%f\n",i,j,s1[1]);
                i=row2; j=row2; printf("xnew[%d][%d]=%f\n",i,j,s2[0]);
                i=row3; j=row1; printf("xnew[%d][%d]=%f\n",i,j,s3[0]);
                i=row3; j=row3; printf("xnew[%d][%d]=%f\n",i,j,s3[1]);
                printf("Sequential Jacobi: Matrix Size = %d; %.1f GFLOPS; Time = %.3f sec; \n",
                        N,13.0*1e-9*N*N*(iter+1)/t,t); 
            } 

            if(rank==rank1)
                MPI_Wait(&requests[0], MPI_STATUSES_IGNORE);
            if(rank==rank2)
                MPI_Wait(&requests[1], MPI_STATUSES_IGNORE);
            if(rank==rank3)
                MPI_Wait(&requests[2], MPI_STATUSES_IGNORE);

            break;
        }

        tmp = old;
        old = new;
        new = tmp;
        if(rank==0 && (iter%nprfreq)==0)
            printf("Iter = %d Resid Norm = %f\n",iter,rhonew);
    }
    return 0;
} 

void init(double **x, double * b)
{ 
    int i,j;

    for(i=0;i<N+2;i++) {
        for(j=0;j<N+2;j++) {
            x[0][i*(N+2)+j]=0.0;
            x[1][i*(N+2)+j]=0.0;
            b[i*(N+2)+j]=i+j; 
        }
    }
}

double rhocalc(double * A)
{ 
    int i, j; 
    double tmp=0.0, S;

    for(i=ibegin;i<iend;i++)
        for(j=1;j<N+1;j++)
            tmp+=A[i*(N+2)+j]*A[i*(N+2)+j];

    MPI_Allreduce(&tmp, &S, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); 
    return(sqrt(S));
}

void update(double **x, double * resid, double * b)
{
    int i, j;
    int offset=2, count=0;
    MPI_Request requests[4];

    if(rank != 0) {
        offset = 0;
        MPI_Isend(&x[old][ibegin * (N+2)], N+2, MPI_DOUBLE, rank-1, 4, MPI_COMM_WORLD, &requests[0]);
        MPI_Irecv(&x[old][(ibegin-1) * (N+2)], N+2, MPI_DOUBLE, rank-1, 4, MPI_COMM_WORLD, &requests[1]);
        count += 2;
    }
    if(rank != size-1) {
        MPI_Isend(&x[old][(iend-1) * (N+2)], N+2, MPI_DOUBLE, rank+1, 4, MPI_COMM_WORLD, &requests[2]);
        MPI_Irecv(&x[old][iend * (N+2)], N+2, MPI_DOUBLE, rank+1, 4, MPI_COMM_WORLD, &requests[3]);
        count += 2;
    }

    MPI_Waitall(count, &requests[offset], MPI_STATUSES_IGNORE);

    for(i=ibegin; i<iend ;i++) {
        for(j=1;j<N+1;j++) {
            x[new][i*(N+2)+j]=b[i*(N+2)+j]-odiag*(x[old][i*(N+2)+j-1]+x[old][i*(N+2)+j+1]+x[old][(i+1)*(N+2)+j]+x[old][(i-1)*(N+2)+j]);
            x[new][i*(N+2)+j]*=recipdiag;
        }
    }

    offset=2, count=0;
    if(rank != 0) {
        offset = 0;
        MPI_Isend(&x[new][ibegin * (N+2)], N+2, MPI_DOUBLE, rank-1, 5, MPI_COMM_WORLD, &requests[0]);
        MPI_Irecv(&x[new][(ibegin-1) * (N+2)], N+2, MPI_DOUBLE, rank-1, 5, MPI_COMM_WORLD, &requests[1]);
        count += 2;
    }
    if(rank != size -1) {
        MPI_Isend(&x[new][(iend-1)*(N+2)], N+2, MPI_DOUBLE, rank+1, 5, MPI_COMM_WORLD, &requests[2]);
        MPI_Irecv(&x[new][iend * (N+2)], N+2, MPI_DOUBLE, rank+1, 5, MPI_COMM_WORLD, &requests[3]);
        count += 2;
    }

    MPI_Waitall(count, &requests[offset], MPI_STATUSES_IGNORE);

    for(i=ibegin;i<iend;i++) {
        for(j=1;j<N+1;j++) {
            resid[i*(N+2)+j]=b[i*(N+2)+j]-diag*x[new][i*(N+2)+j]-odiag*(x[new][i*(N+2)+j+1]+x[new][i*(N+2)+j-1]+x[new][(i-1)*(N+2)+j]+x[new][(i+1)*(N+2)+j]);
        }
    }
} 

double rtclock()
{
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}


