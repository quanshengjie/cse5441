extern "C" __global__ void
mmkernel( float* a, float* b, float* c,
int pitch_a, int pitch_b, int pitch_c,
int n, int m, int p )
{
   int tx = threadIdx.x;
   int i = blockIdx.x*64 + threadIdx.x;
   int j = blockIdx.y;
   __shared__ float cc[32]; 
   float sum0 = 0.0, sum1=0.0;

   for( int ks = 0; ks < p; ks += 32 ){
      cc[tx] = c[(ks+tx) * pitch_c + j];
      __syncthreads();
      for( int k = ks; k < ks+32; ++k ) {
        sum0 += b[i+pitch_b*k] * cc[k-ks];
	sum1 += b[i+32+pitch_b*k] * cc[k-ks];
      }
      __syncthreads();
    }
   
   a[j+pitch_a*i] = sum0;
   a[j+pitch_a*(i+32)] = sum1;
}

