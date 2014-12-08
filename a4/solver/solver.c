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
double clkbegin, clkend;
double t;
int rank;
int size;
double rtclock();

void init(double*,double*,double*);
double rhocalc(double*);
void update(double*,double*,double*,double*); 
void copy(double*,double*);
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
    double rhoinit,rhonew; 
    int i,j,ui;
    int r1, r2, r3;
    int i1, i2, i3;
    MPI_Request requests[6];

    double *b = malloc(sizeof(double)*((N+2)*(N+2)));
    double *xold= malloc(sizeof(double)*((N+2)*(N+2)));
    double *xnew= malloc(sizeof(double)*((N+2)*(N+2)));
    double *resid= malloc(sizeof(double)*((N+2)*(N+2)));

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    init(xold,xnew,b);
    rhoinit = rhocalc(b);

    clkbegin = rtclock();
    for(iter=0;iter<maxiter;iter++){
        update(xold,xnew,resid,b);
        rhonew = rhocalc(resid);
        if(rhonew<eps){
            i1 = (N+2)/4;
            i2 = (N+1)/2;
            i3 = 3*(N+2)/4;
            r1 = i1/(N/size);
            r2 = i2/(N/size);
            r3 = i3/(N/size);

            if(rank==r1) {
                MPI_Isend(&xnew[i1*(N+2)], N+2, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD, &requests[0]);
            }
            if(rank==r2) {
                MPI_Isend(&xnew[i2*(N+2)], N+2, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD, &requests[1]);
            }
            if(rank==r3) {
                MPI_Isend(&xnew[i3*(N+2)], N+2, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD, &requests[2]);
            }

            if(rank == 0) {
                MPI_Irecv(&xnew[i1*(N+2)], N+2, MPI_DOUBLE, r1, 3, MPI_COMM_WORLD, &requests[3]);
                MPI_Irecv(&xnew[i2*(N+2)], N+2, MPI_DOUBLE, r2, 3, MPI_COMM_WORLD, &requests[4]);
                MPI_Irecv(&xnew[i3*(N+2)], N+2, MPI_DOUBLE, r3, 3, MPI_COMM_WORLD, &requests[5]);
                MPI_Waitall(3, &requests[3], MPI_STATUSES_IGNORE);

                clkend = rtclock();
                t = clkend-clkbegin;
                printf("Solution converged in %d iterations\n",iter);
                printf("Final residual norm = %f\n",rhonew);
                printf("Solution at center and four corners of interior N/2 by N/2 grid : \n");
                i=i1; j=(N+2)/4; printf("xnew[%d][%d]=%f\n",i,j,xnew[i*(N+2)+j]);
                i=i1; j=3*(N+2)/4; printf("xnew[%d][%d]=%f\n",i,j,xnew[i*(N+2)+j]);
                i=i2; j=(N+1)/2; printf("xnew[%d][%d]=%f\n",i,j,xnew[i*(N+2)+j]);
                i=i3; j=(N+2)/4; printf("xnew[%d][%d]=%f\n",i,j,xnew[i*(N+2)+j]);
                i=i3; j=3*(N+2)/4; printf("xnew[%d][%d]=%f\n",i,j,xnew[i*(N+2)+j]);
                printf("Sequential Jacobi: Matrix Size = %d; %.1f GFLOPS; Time = %.3f sec; \n",
                        N,13.0*1e-9*N*N*(iter+1)/t,t); 
            } 

            if(rank==r1) {
                MPI_Waitall(1, &requests[0], MPI_STATUSES_IGNORE);
            }
            if(rank==r2) {
                MPI_Waitall(1, &requests[1], MPI_STATUSES_IGNORE);
            }
            if(rank==r3) {
                MPI_Waitall(1, &requests[2], MPI_STATUSES_IGNORE);
            }
            break;
        }
        copy(xold,xnew);
        if((iter%nprfreq)==0 && rank==0)
            printf("Iter = %d Resid Norm = %f\n",iter,rhonew);
    }
    return 0;
} 

void init(double * xold, double * xnew, double * b)
{ 
    int i,j;
    for(i=0;i<N+2;i++){
        for(j=0;j<N+2;j++){
            xold[i*(N+2)+j]=0.0;
            xnew[i*(N+2)+j]=0.0;
            b[i*(N+2)+j]=i+j; 
        }
    }
}

double rhocalc(double * A)
{ 
    double tmp;
    tmp =0.0;
    int id, i, j, nthrds, ibegin, iend, chunk;
    double ptmp;
    id=rank;
    chunk = N/size;
    ibegin = (id*chunk)+1;
    iend = ibegin + chunk;
    ptmp = 0.0;
    for(i=ibegin;i<iend;i++)
        for(j=1;j<N+1;j++)
            ptmp+=A[i*(N+2)+j]*A[i*(N+2)+j];
    MPI_Allreduce(&ptmp, &tmp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); 
    return(sqrt(tmp));
}

void update(double * xold,double * xnew,double * resid, double * b)
{
    int id, i, j, nthrds, ibegin, iend, chunk;
    id=rank;
    chunk = N/size;
    MPI_Request requests[4];

    ibegin = (id*chunk)+1;
    iend = ibegin + chunk;

    if(rank != 0)
    {
        MPI_Isend(&xold[ibegin * (N+2)], N+2, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &requests[0]);
        MPI_Irecv(&xold[(ibegin-1) * (N+2)], N+2, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &requests[1]);
        MPI_Waitall(2, &requests[0], MPI_STATUSES_IGNORE);
    }

    if(rank != size-1)
    {
        MPI_Isend(&xold[(iend-1) * (N+2)], N+2, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &requests[2]);
        MPI_Irecv(&xold[iend * (N+2)], N+2, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &requests[3]);
        MPI_Waitall(2, &requests[2], MPI_STATUSES_IGNORE);
    }
    for(i=ibegin; i<iend ;i++)
    {
        for(j=1;j<N+1;j++)
        {
            xnew[i*(N+2)+j]=b[i*(N+2)+j]-odiag*(xold[i*(N+2)+j-1]+xold[i*(N+2)+j+1]+xold[(i+1)*(N+2)+j]+xold[(i-1)*(N+2)+j]);
            xnew[i*(N+2)+j]*=recipdiag;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if(rank != 0)
    {
        MPI_Isend(&xnew[ibegin * (N+2)], N+2, MPI_DOUBLE, rank-1, 1, MPI_COMM_WORLD, &requests[0]);
        MPI_Irecv(&xnew[(ibegin-1) * (N+2)], N+2, MPI_DOUBLE, rank-1, 1, MPI_COMM_WORLD, &requests[1]);
        MPI_Waitall(2, &requests[0], MPI_STATUSES_IGNORE);
    }
    if(rank != size -1)
    {
        MPI_Isend(&xnew[(iend-1)*(N+2)], N+2, MPI_DOUBLE, rank+1, 1, MPI_COMM_WORLD, &requests[2]);
        MPI_Irecv(&xnew[iend * (N+2)], N+2, MPI_DOUBLE, rank+1, 1, MPI_COMM_WORLD, &requests[3]);
        MPI_Waitall(2, &requests[2], MPI_STATUSES_IGNORE);
    }

    for(i=ibegin;i<iend;i++)
    {
        for(j=1;j<N+1;j++){
            resid[i*(N+2)+j]=b[i*(N+2)+j]-diag*xnew[i*(N+2)+j]-odiag*(xnew[i*(N+2)+j+1]+xnew[i*(N+2)+j-1]+xnew[(i-1)*(N+2)+j]+xnew[(i+1)*(N+2)+j]);
        }
    }
} 

void copy(double * xold, double * xnew)
{
    int id, i, j, nthrds, ibegin, iend, chunk;
    id=rank;
    chunk = N/size;
    ibegin = (id*chunk)+1;
    iend = ibegin + chunk;

    for(i=ibegin;i<iend;i++)
        for(j=1;j<N+1;j++)
            xold[i*(N+2)+j]=xnew[i*(N+2)+j];
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


