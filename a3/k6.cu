extern "C" __global__ void
mmkernel( float* a, float* b, float* c,
        int pitch_a, int pitch_b, int pitch_c,
        int n, int m, int p )
{
    int i = blockIdx.x*32 + threadIdx.x;
    int j = blockIdx.y;
    float sum = 0.0;
    for( int k = 0; k < p; ++k )
        sum += b[k*pitch_b+i] * c[j+pitch_c*k];
    a[j+pitch_a*i] = sum;
}
