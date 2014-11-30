extern "C" __global__ void
mmkernel( float* a, float* b, float* c,
int pitch_a, int pitch_b, int pitch_c,
int n, int m, int p )
{
   int tx = threadIdx.x;
   int i = blockIdx.x*32 + threadIdx.x;
   int j = blockIdx.y*2;
   __shared__ float cc0[32], cc1[32]; 
   float sum0 = 0.0, sum1 = 0.0;
   
   for( int ks = 0; ks < p; ks += 32 ){
      cc0[tx] = c[(ks+tx) * pitch_c + j];
      cc1[tx] = c[(ks+tx) * pitch_c + j+1];
      __syncthreads();
      for( int k = ks; k < ks+32; ++k ) {
        float rb = b[i+pitch_b*k];
	sum0 += rb * cc0[k-ks];
	sum1 += rb * cc1[k-ks];
      }
      __syncthreads();
    }
   
   a[j+pitch_a*i] = sum0;
   a[j+1+pitch_a*i] = sum1;
}

