#include <unistd.h>
#include <stdio.h>
#include <sys/time.h>
#define N 4096
#define Niter 10
#define threshold 0.0000001

double A[N][N], x[N],y[N],z[N],y1[N],z1[N];
main(){
double rtclock();
void pa1p1(int n, int niter, double a[][n], double p[n], double q[n], double r[n]);
void pa1p1opt(int n, int niter, double a[][n], double p[n], double q[n], double r[n]);
void compare(int n, double wref[n], double w[n]);

double clkbegin, clkend;
double t;

int i,j,it;

  for(i=0;i<N;i++)
   { 
     x[i] = i; 
     y[i]= 0; z[i] = 1.0;
     y1[i]= 0; z1[i] = 1.0;
     for(j=0;j<=N;j++) A[i][j] = (i+2*j)/(2*N);
   }

   pa1p1(N,Niter,A,x,y,z);
   pa1p1opt(N,Niter,A,x,y1,z1);
   compare(N,y,y1);

}

void pa1p1(int n, int niter, double a[][n], double p[n], double q[n], double r[n])
{ int i,j,it;
  double clkbegin, clkend,t;
  double rtclock();
  clkbegin = rtclock();
// Outer "it" loop to increase timing granularity
  for(it=0;it<niter;it++)
     for(i=0;i<n;i++)
        for(j=0;j<n;j++)
          {
           q[i] = q[i] + a[i][j]*p[j];
           r[i] = r[i] + a[j][i]*p[j];
          }
  clkend = rtclock();
  t = clkend-clkbegin;
// To foil compiler optimizer - "dead code" elimination
  if (q[n/2]*q[n/2] < -100.0) printf("%f\n",q[n/2]);
  printf("Problem 1 Reference Version: Matrix Size = %d; %.1f GFLOPS; Time = %.3f sec; \n",
          n,4.0*1e-9*n*n*niter/t,t);
}

void pa1p1opt(int n, int niter, double a[][n], double p[n], double q[n], double r[n])
{ int i,j,it;
  double clkbegin, clkend,t;
  double rtclock();
  clkbegin = rtclock();
// Initially identical to reference version. 
// Replace it with optimized version
  for(it=0;it<niter;it++)
  {
    for(i=0;i<n;i++)
      for(j=0;j<n;j++)
       {
        q[i] = q[i] + a[i][j]*p[j];
        r[i] = r[i] + a[j][i]*p[j];
       }
  }
  clkend = rtclock();
  t = clkend-clkbegin;
// To foil compiler optimizer - "dead code" elimination
  if (q[n/2]*q[n/2] < -100.0) printf("%f\n",q[n/2]);
  printf("Problem 1 Optimized version: %d; %.1f GFLOPS; Time = %.3f sec; \n",
          n,4.0*1e-9*n*n*niter/t,t);
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

void compare(int n, double wref[n], double w[n])
{
double maxdiff,this_diff;
int numdiffs;
int i;
  numdiffs = 0;
  maxdiff = 0;
  for (i=0;i<n;i++)
    {
     this_diff = wref[i]-w[i];
     if (this_diff < 0) this_diff = -1.0*this_diff;
     if (this_diff>threshold)
      { numdiffs++;
        if (this_diff > maxdiff) maxdiff=this_diff;
      }
    }
   if (numdiffs > 0)
      printf("%d Diffs found over threshold %f; Max Diff = %f\n",
               numdiffs,threshold,maxdiff);
   else
      printf("No differences found between base and test versions\n");
}
