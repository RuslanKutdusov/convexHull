#include <stdio.h>


#define CUDA_CHECK_RETURN( value ) {                                            \
    cudaError_t _m_cudaStat = value;                                        \
    if ( _m_cudaStat != cudaSuccess ) {                                     \
        fprintf( stderr, "Error '%s' at line %d in file %s\n",                  \
                cudaGetErrorString( _m_cudaStat ), __LINE__, __FILE__ );        \
        exit( 1 );                                                          \
    } }


//
void bandwidthTest( int gpu1, int gpu2, size_t memSize )
{
    printf( "Bandwidth test between " );
    float measuredTime = 0.0f;
    float bandwidthPeerAccessDisabled = 0.0f;
    float bandwidthPeerAccessEnabled = 0.0f;
    cudaEvent_t start, stop;
    cudaDeviceProp deviceProp;

    // alloc host memory
    unsigned char* h_array = new unsigned char[ memSize ];
    for( size_t i = 0; i < memSize; i++ )
        h_array[ i ] = ( unsigned char )( i & 0xff );

    // set first device
    CUDA_CHECK_RETURN( cudaSetDevice( gpu1 ) );

    // get gpu1 name
    CUDA_CHECK_RETURN( cudaGetDeviceProperties( &deviceProp, gpu1 ) );
    printf("GPU%d %s and", gpu1, deviceProp.name );

    // create events
    CUDA_CHECK_RETURN( cudaEventCreate( &start ) );
    CUDA_CHECK_RETURN( cudaEventCreate( &stop ) );

    // alloc mem on gpu1
    unsigned char *d_gpu1Array;
    CUDA_CHECK_RETURN( cudaMalloc( ( void ** )&d_gpu1Array, memSize ) );
    CUDA_CHECK_RETURN( cudaMemcpy( d_gpu1Array, h_array, memSize, cudaMemcpyHostToDevice ) );

    // set second gpu
    CUDA_CHECK_RETURN( cudaSetDevice( gpu2 ) );

    // get gpu2 name
    CUDA_CHECK_RETURN( cudaGetDeviceProperties( &deviceProp, gpu1 ) );
    printf(" GPU%d %s\n", gpu2, deviceProp.name );

    // alloc mem on gpu2
    unsigned char *d_gpu2Array;
    CUDA_CHECK_RETURN( cudaMalloc( ( void ** )&d_gpu2Array, memSize ) );

    // check accessible
    int accessible = 0;
    cudaDeviceCanAccessPeer( &accessible, gpu2, gpu1 );
    printf( "   GPU%d is capable of directly accessing memory from GPU%d: %s\n", gpu2, gpu1, accessible ? "yes" : "no" );
    for( int i = 0; i < 1 + accessible; i++ )
    {
        CUDA_CHECK_RETURN( cudaSetDevice( gpu2 ) );
        if( i == 1 )
            CUDA_CHECK_RETURN( cudaDeviceEnablePeerAccess( gpu1, 0 ) );

        // return back gpu1, cause create events on it
        CUDA_CHECK_RETURN( cudaSetDevice( gpu1 ) );

        CUDA_CHECK_RETURN( cudaEventRecord( start, 0 ) );

        const size_t STEPS = 10;
        for( size_t j = 0; j < STEPS; j++ )
            CUDA_CHECK_RETURN( cudaMemcpyPeer( d_gpu2Array, gpu2, d_gpu1Array, gpu1, memSize ) );

        CUDA_CHECK_RETURN( cudaEventRecord( stop, 0 ) );

        // waiting
        CUDA_CHECK_RETURN( cudaSetDevice( gpu2 ) );
        CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
        CUDA_CHECK_RETURN( cudaSetDevice( gpu1 ) );
        CUDA_CHECK_RETURN( cudaDeviceSynchronize() );

        //
        CUDA_CHECK_RETURN( cudaEventElapsedTime( &measuredTime, start, stop ) );

        //
        ( i == 0 ? bandwidthPeerAccessDisabled : bandwidthPeerAccessEnabled ) = ( 1e3f * memSize * ( float )STEPS) / ( measuredTime * ( float )( 1 << 20 ) );
    }

    printf( "   Bandwidth with enabled peer acces:  %f mb/s\n", bandwidthPeerAccessEnabled );
    printf( "   Bandwidth with disabled peer acces: %f mb/s\n", bandwidthPeerAccessDisabled );

    //
    free( h_array );
    CUDA_CHECK_RETURN( cudaEventDestroy( stop ) );
    CUDA_CHECK_RETURN( cudaEventDestroy( start ) );
    CUDA_CHECK_RETURN( cudaFree( d_gpu1Array ) );

    CUDA_CHECK_RETURN( cudaSetDevice( gpu2 ) );
    CUDA_CHECK_RETURN( cudaFree( d_gpu2Array) );
    if( accessible )
        CUDA_CHECK_RETURN( cudaDeviceDisablePeerAccess( gpu1 ) );
}


//
int main( int argc, char* argv[] )
{
    size_t memSize = 1 << 29;
    if( argc == 2 )
         memSize = atoi( argv[ 1 ] );

    int deviceCount;
    CUDA_CHECK_RETURN( cudaGetDeviceCount( &deviceCount ) );
    printf( "Device count: %d %u\n", deviceCount, memSize );

    for( int i = 0; i < deviceCount; i++ )
        for( int j = i + 1; j < deviceCount; j++ )
        {
            if( i == j )
                continue;

            printf( "\n" );
            bandwidthTest( i, j, memSize );
        }

    return 0;
}