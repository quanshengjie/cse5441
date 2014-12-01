extern "C" __global__ void
mmkernel( float* a, float* b, float* c,
    int pitch_a, int pitch_b, int pitch_c,
    int n, int m, int p )
{
    int bx = blockDim.x;
    int i = blockIdx.x * bx + threadIdx.x;
    int j = blockIdx.y;
    float sum = 0.0;

    for( int k = 0; k < p; ++k )
        sum += b[k*pitch_b+i] * c[k*pitch_c+j];
    a[j+pitch_a*i] = sum;
}

