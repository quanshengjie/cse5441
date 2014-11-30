extern "C" __global__ void
mmkernel( float* a, float* b, float* c,
int pitch_a, int pitch_b, int pitch_c,
int n, int m, int p )
{
   int tx = threadIdx.x;
   int i = blockIdx.x*32 + threadIdx.x;
   int j = blockIdx.y;
   __shared__ float cc[32]; 
   float sum = 0.0;
   
   
   for( int ks = 0; ks < p; ks += 32 ){
      cc[tx] = c[(ks+tx) * pitch_c + j];
      __syncthreads();
      for( int k = ks; k < ks+32; ++k )
        sum += b[i+pitch_b*k] * cc[k-ks];
    }
   
   a[j+pitch_a*i] = sum;
}

