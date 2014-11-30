extern "C" __global__ void
mmkernel( float* a, float* b, float* c,
  int pitch_a, int pitch_b, int pitch_c,
  int n, int m, int p )
{
    int tx = threadIdx.x;
    int bx = blockDim.x;
    int i = blockIdx.x * bx * 2 + tx;
    int j = blockIdx.y;
    __shared__ float cb[512];

    float sum0 = 0.0, sum1 = 0.0;
    for( int ks = 0; ks < p; ks += bx ){
      cb[tx] = c[ks+tx+pitch_c*j];
      __syncthreads();
      for( int k = ks; k < ks+bx; ++k ){
        sum0 += b[i+pitch_b*k] * cb[k-ks];
        sum1 += b[i+bx+pitch_b*k] * cb[k-ks];
      }
      __syncthreads();
    }
    a[i+pitch_a*j] = sum0;
    a[i+bx+pitch_a*j] = sum1;
}
