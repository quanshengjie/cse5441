extern "C" __global__ void
mmkernel( float* a, float* b, float* c,
    int pitch_a, int pitch_b, int pitch_c,
    int n, int m, int p )
{
    int tx = threadIdx.x;
    int bx = blockDim.x;
    int i = blockIdx.x*bx*2 + threadIdx.x;
    int j = blockIdx.y;
    __shared__ float cc[512]; 
    float sum0 = 0.0, sum1=0.0;

    for( int ks = 0; ks < p; ks += bx ){
        cc[tx] = c[(ks+tx) * pitch_c + j];
        __syncthreads();
        for( int k = ks; k < ks+bx; ++k ) {
            sum0 += b[i+pitch_b*k] * cc[k-ks];
            sum1 += b[i+bx+pitch_b*k] * cc[k-ks];
        }
        __syncthreads();
    }

    a[j+pitch_a*i] = sum0;
    a[j+pitch_a*(i+bx)] = sum1;
}

