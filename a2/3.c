#include <math.h>
#include <stdio.h>
#include <sys/time.h>
#include <omp.h>

#define N 1024
#define diag 5.0
#define recipdiag 0.2
#define odiag -1.0
#define eps  1.0E-6
#define maxiter 10000
#define nprfreq 10

double clkbegin, clkend;
double t;
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
 double b[(N+2)*(N+2)];
 double xold[(N+2)*(N+2)];
 double xnew[(N+2)*(N+2)];
 double resid[(N+2)*(N+2)];
 double rhoinit,rhonew; 
 int i,j,iter,u;

 omp_set_num_threads(atoi(argv[1]));

 init(xold,xnew,b);
 rhoinit = rhocalc(b);

  clkbegin = rtclock();
 for(iter=0;iter<maxiter;iter++){
  update(xold,xnew,resid,b);
  rhonew = rhocalc(resid);
  if(rhonew<eps){
   clkend = rtclock();
   t = clkend-clkbegin;
   printf("Solution converged in %d iterations\n",iter);
   printf("Final residual norm = %f\n",rhonew);
   printf("Solution at center and four corners of interior N/2 by N/2 grid : \n");
   i=(N+2)/4; j=(N+2)/4; printf("xnew[%d][%d]=%f\n",i,j,xnew[i*(N+2)+j]);
   i=(N+2)/4; j=3*(N+2)/4; printf("xnew[%d][%d]=%f\n",i,j,xnew[i*(N+2)+j]);
   i=(N+1)/2; j=(N+1)/2; printf("xnew[%d][%d]=%f\n",i,j,xnew[i*(N+2)+j]);
   i=3*(N+2)/4; j=(N+2)/4; printf("xnew[%d][%d]=%f\n",i,j,xnew[i*(N+2)+j]);
   i=3*(N+2)/4; j=3*(N+2)/4; printf("xnew[%d][%d]=%f\n",i,j,xnew[i*(N+2)+j]);
   printf("Sequential Jacobi: Matrix Size = %d; %.1f GFLOPS; Time = %.3f sec; \n",
          N,13.0*1e-9*N*N*(iter+1)/t,t); 
   break;
  } 
  copy(xold,xnew);
  if((iter%nprfreq)==0)
    printf("Iter = %d Resid Norm = %f\n",iter,rhonew);
 }
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
 double S = 0.0; 
 #pragma omp parallel
 {
 int rank, size, istart, iend, chunk;
 int i, j;
 double tmp = 0.0;
 rank=omp_get_thread_num();
 size = omp_get_num_threads();
 chunk = N/size;
 istart = (rank*chunk)+1;
 iend = istart + chunk;
 for(i=istart; i<iend; i++) {
   for(j=1;j<N+1;j++) {
     tmp+=A[i*(N+2)+j]*A[i*(N+2)+j];
   }
 }
 #pragma omp atomic
 S += tmp;
 }
 return(sqrt(S));
}

void update(double * xold,double * xnew,double * resid, double * b)
{
 #pragma omp parallel
 {
 int rank, size, istart, iend, chunk;
 int i, j;
 rank=omp_get_thread_num();
 size = omp_get_num_threads();
 chunk = N/size;
 istart = (rank*chunk)+1;
 iend = istart + chunk;
 for(i=istart; i<iend; i++) {
   for(j=1;j<N+1;j++) {
     xnew[i*(N+2)+j]=b[i*(N+2)+j]-odiag*(xold[i*(N+2)+j-1]+xold[i*(N+2)+j+1]+xold[(i+1)*(N+2)+j]+xold[(i-1)*(N+2)+j]);
     xnew[i*(N+2)+j]*=recipdiag;
   }
 }
 }

 #pragma omp parallel
 {
 int rank, size, istart, iend, chunk;
 int i, j;
 rank=omp_get_thread_num();
 size = omp_get_num_threads();
 chunk = N/size;
 istart = (rank*chunk)+1;
 iend = istart + chunk;
 for(i=istart; i<iend; i++) {
   for(j=1;j<N+1;j++) {
     resid[i*(N+2)+j]=b[i*(N+2)+j]-diag*xnew[i*(N+2)+j]-odiag*(xnew[i*(N+2)+j+1]+xnew[i*(N+2)+j-1]+xnew[(i-1)*(N+2)+j]+xnew[(i+1)*(N+2)+j]);
   }
 } 
 }
} 
  
void copy(double * xold, double * xnew)
{ 
 #pragma omp parallel
 {
 int rank, size, istart, iend, chunk;
 int i, j;
 rank=omp_get_thread_num();
 size = omp_get_num_threads();
 chunk = N/size;
 istart = (rank*chunk)+1;
 iend = istart + chunk;
 for(i=istart; i<iend; i++) {
  for(j=1;j<N+1;j++) {
    xold[i*(N+2)+j]=xnew[i*(N+2)+j];
  }
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


