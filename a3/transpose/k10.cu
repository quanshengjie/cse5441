extern "C" __global__ void
mmkernel( float* a, float* b, float* c,
    int pitch_a, int pitch_b, int pitch_c,
    int n, int m, int p )
{
    int tx = threadIdx.x;
    int bx = blockDim.x;
    int i = blockIdx.x*bx*2 + threadIdx.x;
    int j = blockIdx.y*2;
    __shared__ float cc0[512], cc1[512]; 
    float sum0 = 0.0, sum1=0.0, sum2 =0.0, sum3=0.0;

    for( int ks = 0; ks < p; ks += bx ){
        cc0[tx] = c[(ks+tx) * pitch_c + j];
        cc1[tx] = c[(ks+tx) * pitch_c + j+1];
        __syncthreads();
        for( int k = ks; k < ks+bx; ++k ) {
            sum0 += b[i+pitch_b*k] * cc0[k-ks];
            sum1 += b[i+bx+pitch_b*k] * cc0[k-ks];
            sum2 += b[i+pitch_b*k] * cc1[k-ks];
            sum3 += b[i+bx+pitch_b*k] * cc1[k-ks];
        }
        __syncthreads();
    }

    a[j+pitch_a*i] = sum0;
    a[j+pitch_a*(i+bx)] = sum1;
    a[j+1+pitch_a*i] = sum2;
    a[j+1+pitch_a*(i+bx)] = sum3;
}

