#include <stdio.h>
#include <unistd.h>

#define CUDA_CHECK_RETURN( value ) {											\
	cudaError_t _m_cudaStat = value;										\
	if ( _m_cudaStat != cudaSuccess ) {										\
		fprintf( stderr, "Error '%s' at line %d in file %s\n",					\
				cudaGetErrorString( _m_cudaStat ), __LINE__, __FILE__ );		\
		exit( 1 );															\
	} }

int main()
{
	int deviceCount = 0;
	CUDA_CHECK_RETURN( cudaGetDeviceCount( &deviceCount ) );
	printf( "Device count: %d\n", deviceCount );

	// 
	for( int i = 0; i < deviceCount; i++ )
	{
		CUDA_CHECK_RETURN( cudaSetDevice( i ) );
		cudaDeviceProp deviceProp;
		CUDA_CHECK_RETURN( cudaGetDeviceProperties( &deviceProp, i ) );
		printf("GPU%d is capable of directly accessing memory from \n", i );
		
		for( int j = 0; j < deviceCount; j++ )
		{
			if( i == j )
				continue;

			int accessible;
			cudaDeviceCanAccessPeer( &accessible, i, j );
			printf( "	GPU%d: %s\n", j, accessible ? "yes" : "no" );
		}
	}

	return 0;
}
