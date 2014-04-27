#include "gpu.hpp"

#include <stdio.h>
#include <stdint.h>
#include <vector>

#define CUDA_CHECK_RETURN( value ) {											\
	cudaError_t _m_cudaStat = value;										\
	if ( _m_cudaStat != cudaSuccess ) {										\
		fprintf( stderr, "Error '%s' at line %d in file %s\n",					\
				cudaGetErrorString( _m_cudaStat ), __LINE__, __FILE__ );		\
		exit( 1 );															\
	} }

namespace gpu
{


//
const int MAX_GPU_COUNT = 8;
const uint32_t BLOCK_DIM = 192;


//
__global__ void kernel1( FP* hyperplanes, FP* points, uint32_t n, uint32_t dimX, uint32_t numberOfHyperplanes, uint32_t numberOfPoints )
{
	if( blockIdx.x * blockDim.x + threadIdx.x >= numberOfHyperplanes )
		return;

	uint32_t offsetToHyperplanesChunk = blockIdx.x * blockDim.x * ( n + 1 );

	FP resultDistance = 0.0;
	for( uint32_t k = 0; k < numberOfPoints * n; k += n * BLOCK_DIM )
		#pragma unroll
		for( uint32_t i = 0; i < BLOCK_DIM; i++ )
		{
			FP d = 0.0;

			uint32_t offsetToPoint = k + i;

			// dot product of point and normal is distance
			uint16_t j = 0;
			for( ; j < dimX * BLOCK_DIM; j += BLOCK_DIM ) 
				d += points[ offsetToPoint + j ] * hyperplanes[ offsetToHyperplanesChunk + threadIdx.x + j ];
			d += points[ offsetToPoint + j ] * hyperplanes[ offsetToHyperplanesChunk + threadIdx.x + j ]; 

			if( d > resultDistance )
				resultDistance = d;
		}
 
    hyperplanes[ offsetToHyperplanesChunk + threadIdx.x + n * BLOCK_DIM ] = resultDistance;
}


//
__global__ void kernel1_1( FP** hyperplanes, int32_t deviceCount, uint32_t n, uint32_t numberOfHyperplanes )
{
	if( blockIdx.x * blockDim.x + threadIdx.x >= numberOfHyperplanes )
		return;

	uint32_t offset = blockIdx.x * blockDim.x * ( n + 1 ) + threadIdx.x + n * BLOCK_DIM;

	FP resultDistance = hyperplanes[ 0 ][ offset ];
	for( int32_t i = 1; i < deviceCount; i++ )
	{
		if( hyperplanes[ i ][ offset ] > resultDistance )
			resultDistance = hyperplanes[ i ][ offset ];
	}
	hyperplanes[ 0 ][ offset ] = resultDistance;
}


// sending dimX as argument is to reduce registers usage
__global__ void kernel2( FP* hyperplanes, FP* points, uint32_t n, uint32_t dimX, uint32_t numberOfHyperplanes, uint32_t numberOfPoints )
{
	if( blockIdx.x * blockDim.x + gridDim.x * blockDim.x * blockIdx.y + threadIdx.x >= numberOfPoints )
		return;

	uint32_t offsetToPointsChunk = ( blockIdx.x * blockDim.x + gridDim.x * blockDim.x * blockIdx.y ) * n;

	FP funcVal = points[ offsetToPointsChunk + threadIdx.x + ( n - 1 ) * BLOCK_DIM ];
	FP convexVal = funcVal;

	for( uint32_t i = 0; i < numberOfHyperplanes; i++ )
	{
		uint32_t offsetToHyperplane = ( i - i % BLOCK_DIM ) * ( n + 1 ) + ( i % BLOCK_DIM );

		FP val = 0.0;
		// xi - iter->first
		// Ni - hyperplane normal
		// val = x(n - 1) = ( -N0*x0 - N1*x1 - ... - N(n - 2)*x(n - 2) + xn ) / N(n - 1)
		for( uint8_t j = 0; j < dimX; j++ )
			val -= points[ offsetToPointsChunk + threadIdx.x + j * BLOCK_DIM ] * hyperplanes[ offsetToHyperplane + j * BLOCK_DIM ];
		val += hyperplanes[ offsetToHyperplane + n * BLOCK_DIM ];
		val /= hyperplanes[ offsetToHyperplane + ( n - 1 ) * BLOCK_DIM ] + EPSILON;

		if( i == 0 )
		{
			convexVal = val;
			continue;
		}

		if( val < convexVal && val >= funcVal )
			convexVal = val;
	}


	points[ offsetToPointsChunk + threadIdx.x + ( n - 1 ) * BLOCK_DIM ] = convexVal;
}


//
void getGridAndBlockDim( int n, dim3& gridDim, dim3& blockDim, int device )
{
	// gpu hardware limits for compute caps 1.2 and 2.0
	const int warpSize = 32;
	const int blocksPerSM = 8;

	cudaDeviceProp deviceProp;
	CUDA_CHECK_RETURN( cudaGetDeviceProperties( &deviceProp, device ) );

	int warpCount = ( n / warpSize ) + ( ( ( n % warpSize ) == 0 ) ? 0 : 1 );

	int threadsPerBlock = deviceProp.maxThreadsPerMultiProcessor / blocksPerSM;
	int warpsPerBlock = threadsPerBlock / warpSize;

	int blockCount = ( warpCount / warpsPerBlock ) + ( ( ( warpCount % warpsPerBlock ) == 0 ) ? 0 : 1 );

	blockDim = dim3( threadsPerBlock, 1, 1 );

	gridDim = dim3( blockCount, 1, 1 );

	if( blockCount > deviceProp.maxGridSize[ 0 ] )
	{
		gridDim.x = gridDim.y = sqrt( blockCount );
		if( gridDim.x * gridDim.x < blockCount )
			gridDim.x += 1;
	}

	printf( "GPU%d: %s, Task size: %d, warp number: %d, threads per block: %d, warps per block: %d, grid: (%d, %d, 1)\n", device, deviceProp.name, n, warpCount, threadsPerBlock, warpsPerBlock, gridDim.x, gridDim.y );
}


//
__host__ void makeConvex( ScalarFunction& func, const uint32_t& dimX, const uint32_t& numberOfPoints )
{
	if( dimX == 0 )
		return;
	
	FP dFi = PI / ( numberOfPoints - 1 );

	uint32_t n = dimX + 1; // space dimension

	uint32_t numberOfHyperplanes = pow( numberOfPoints, n - 1 );

	// first x0.. x(n - 2) elements are independent vars. in 2D it will be x
	// x(n - 1) element dependent var. . in 2D it will be y
	// xn - constant, represents distance between O and hyperplane
	uint32_t hyperplanesNumInArray = numberOfHyperplanes;
	uint32_t mod = hyperplanesNumInArray % BLOCK_DIM;
	hyperplanesNumInArray += ( mod == 0 ) ? 0 : BLOCK_DIM - mod;
	uint32_t hyperplanesArraySize = hyperplanesNumInArray * ( n + 1 );
	FP* hyperplanes = new FP[ hyperplanesArraySize ];

	uint32_t pointsNumInArray = func.size();
	mod = pointsNumInArray % BLOCK_DIM;
	pointsNumInArray += ( mod == 0 ) ? 0 : BLOCK_DIM - mod;
	uint32_t pointsArraySize = pointsNumInArray * n;
	FP* points = new FP[ pointsArraySize ];

	printf( "Memory allocated for hyperplanes: %u %u\n", hyperplanesArraySize * sizeof( FP ), numberOfHyperplanes );
	printf( "Memory allocated for points: %u %u %u\n", pointsArraySize * sizeof( FP ), pointsArraySize, func.size() );

	{
		uint32_t i = 0;
		for( ScalarFunction::iterator iter = func.begin(); iter != func.end(); ++iter, i++ )
		{
			uint32_t offsetToPoint = ( i - i % BLOCK_DIM ) * n + ( i % BLOCK_DIM );

			for( uint32_t j = 0; j < dimX; j++ )
				points[ offsetToPoint + j * BLOCK_DIM ] = iter->first[ j ];

			points[ offsetToPoint + ( n - 1 ) * BLOCK_DIM ] = iter->second;
		}

		for( ; i < pointsNumInArray; i++ )
		{
			uint32_t offsetToPoint = ( i - i % BLOCK_DIM ) * n + ( i % BLOCK_DIM );

			for( uint32_t j = 0; j < dimX; j++ )
				points[ offsetToPoint + j * BLOCK_DIM ] = 0.0;

			points[ offsetToPoint + ( n - 1 ) * BLOCK_DIM ] = 0.0;
		}
	}

	FPVector fi( dimX, 0.0 );

	for( uint32_t i = 0; i < numberOfHyperplanes; i++ )
	{
		uint32_t offset = ( i - i % BLOCK_DIM ) * ( n + 1 ) + ( i % BLOCK_DIM );

		for( uint32_t j = 0; j < n; j++ )
		{
			FP* normalComponent = &hyperplanes[ offset + j * BLOCK_DIM ];

			*normalComponent = 1.0;
			for( uint32_t k = 0; k < j; k++ )
				*normalComponent *= sin( fi[ k ] );

			if( j != n - 1 )
				*normalComponent *= cos( fi[ j ] );
		}

		// not good enough
		bool shift = true;
		for( uint32_t k = 0; ( k < dimX ) && shift; k++ )
		{
			if( fabs( fi[ k ] - PI ) <= EPSILON )
			{
				fi[ k ] = 0.0;
				shift = true;	
			}
			else
			{
				fi[ k ] += dFi;
				shift = false;
			}

			if( fi[ k ] - PI > EPSILON )
				fi[ k ] = PI;
		}
	}


	//
	int deviceCount = 0;
	CUDA_CHECK_RETURN( cudaGetDeviceCount( &deviceCount ) );
	printf( "Available device count: %d\n", deviceCount );
	if( deviceCount > MAX_GPU_COUNT )
	{
		printf( "Too much GPUs %d\n", deviceCount );
		deviceCount = MAX_GPU_COUNT;
	}

	// особенность суперкомпьютера Уран, 8 видеокарт одного узла по сути разбиты на 2 части
	// такие, что для видеокарт одной части возможен peer access, но для видеокарт из разных частей - нет.
	// определяем для каких видеокарт возможен peer access( они разбиваются на части(группы) ). 
	// выбираем ту группу для работы, кол-во видеокарт в которой больше, чем в другой.
	std::vector< int > devicesGroups[ 2 ];

	// 
	{
		CUDA_CHECK_RETURN( cudaSetDevice( 0 ) );
		devicesGroups[ 0 ].push_back( 0 );
		for( int j = 1; j < deviceCount; j++ )
		{
			int accessible;
			cudaDeviceCanAccessPeer( &accessible, j, 0 );
			if( accessible )
				devicesGroups[ 0 ].push_back( j );
			else
				devicesGroups[ 1 ].push_back( j );
		}
	}

	std::vector< int >& usedDevices = ( devicesGroups[ 0 ].size() > devicesGroups[ 1 ].size() ) ? devicesGroups[ 0 ] : devicesGroups[ 1 ];

	deviceCount = usedDevices.size();

	// enabling peer access
	CUDA_CHECK_RETURN( cudaSetDevice( usedDevices[ 0 ] ) );
	for( int i = 1; i < ( int )usedDevices.size(); i++ )	
	{
		CUDA_CHECK_RETURN( cudaDeviceEnablePeerAccess( usedDevices[ i ], 0 ) );
	}

	printf( "Used device count: %d\n", deviceCount );

	const uint32_t pointsChunksNumber = pointsNumInArray / BLOCK_DIM;
	const uint32_t pointsChunksPerDevice = pointsChunksNumber / deviceCount;
	const uint32_t pointsChunksForLastDevice = pointsChunksNumber - pointsChunksPerDevice * ( deviceCount - 1 );
	#define CALC_POINT_NUMBER_PER_DEVICE int pointsPerCurrentDevice = ( ( i == deviceCount - 1 ) ? pointsChunksForLastDevice : pointsChunksPerDevice ) * BLOCK_DIM;
	FP* d_hyperplanes[ MAX_GPU_COUNT ];
	FP* d_points[ MAX_GPU_COUNT ];
	dim3 gridDim, blockDim;

	//
	printf( "Memory preparing...\n" );
	for( int i = 0; i < deviceCount; i++ )
	{
		int device = usedDevices[ i ];
		CUDA_CHECK_RETURN( cudaSetDevice( device ) );
		// optimization
		CUDA_CHECK_RETURN( cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 ) );

		//
		CUDA_CHECK_RETURN( cudaMalloc( &d_hyperplanes[ i ], hyperplanesArraySize * sizeof( FP ) ) );

		if( i == 0 )
		{
			CUDA_CHECK_RETURN( cudaMemcpy( d_hyperplanes[ i ], hyperplanes, hyperplanesArraySize * sizeof( FP ), cudaMemcpyHostToDevice ) );
		}
		else
		{
			// TODO: smart copying, pair
			int lastDevice = usedDevices[ i - 1 ];
			CUDA_CHECK_RETURN( cudaMemcpyPeer( d_hyperplanes[ i ], device, d_hyperplanes[ i - 1 ], lastDevice, hyperplanesArraySize * sizeof( FP ) ) );
		}

		int arrayOffset = pointsChunksPerDevice * BLOCK_DIM * i * n;
		CALC_POINT_NUMBER_PER_DEVICE

		//
		int bytesCount = pointsPerCurrentDevice * n * sizeof( FP );
		CUDA_CHECK_RETURN( cudaMalloc( &d_points[ i ], bytesCount ) );
		CUDA_CHECK_RETURN( cudaMemcpy( d_points[ i ], points + arrayOffset, bytesCount, cudaMemcpyHostToDevice ) );
	}

	//
	printf( "Running first kernel...\n" );
	for( int i = 0; i < deviceCount; i++ )
	{
		int device = usedDevices[ i ];
		CUDA_CHECK_RETURN( cudaSetDevice( device ) );	

		CALC_POINT_NUMBER_PER_DEVICE

		getGridAndBlockDim( numberOfHyperplanes, gridDim, blockDim, device );
		kernel1<<< gridDim, blockDim >>>( d_hyperplanes[ i ], d_points[ i ], n, dimX, numberOfHyperplanes, pointsPerCurrentDevice );
	}

	//
	printf( "Synchronizing...\n" );
	for( int i = 0; i < deviceCount; i++ )
	{
		int device = usedDevices[ i ];
		CUDA_CHECK_RETURN( cudaSetDevice( device ) );
		CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
		CUDA_CHECK_RETURN( cudaGetLastError() );
	}

	// no need to run next kernel if device only one
	if( deviceCount > 1 )
	{
		//
		printf( "Running second kernel...\n" );

		int device = usedDevices[ 0 ];
		CUDA_CHECK_RETURN( cudaSetDevice( device ) );

		FP** hostAllocatedMem;
		cudaHostAlloc( ( void** )&hostAllocatedMem, deviceCount * sizeof( FP* ), cudaHostAllocDefault );
		for( int i = 0; i < deviceCount; i++ )
			hostAllocatedMem[ i ] = d_hyperplanes[ i ];

		getGridAndBlockDim( numberOfHyperplanes, gridDim, blockDim, device );
		kernel1_1<<< gridDim, blockDim >>>( hostAllocatedMem, deviceCount, n, numberOfHyperplanes );

		printf( "Synchronizing...\n" );
		CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
		CUDA_CHECK_RETURN( cudaGetLastError() );
		CUDA_CHECK_RETURN( cudaFreeHost( hostAllocatedMem ) );
	}

	//
	printf( "Running third kernel...\n" );
	for( int i = 0; i < deviceCount; i++ )
	{
		int device = usedDevices[ i ];
		CUDA_CHECK_RETURN( cudaSetDevice( device ) );

		// copy hyperplanes from first device to others
		if( i != 0 )
		{
			// TODO: smart copying, pair
			int lastDevice = usedDevices[ i - 1 ];
			CUDA_CHECK_RETURN( cudaMemcpyPeer( d_hyperplanes[ i ], device, d_hyperplanes[ i - 1 ], lastDevice, hyperplanesArraySize * sizeof( FP ) ) );
		}

		CALC_POINT_NUMBER_PER_DEVICE

		getGridAndBlockDim( pointsPerCurrentDevice, gridDim, blockDim, device );
		kernel2<<< gridDim, blockDim >>>( d_hyperplanes[ i ], d_points[ i ], n, dimX, numberOfHyperplanes, pointsPerCurrentDevice );
	}

	//
	printf( "Synchronizing...\n" );
	for( int i = 0; i < deviceCount; i++ )
	{
		int device = usedDevices[ i ];
		CUDA_CHECK_RETURN( cudaSetDevice( device ) );
		CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
		CUDA_CHECK_RETURN( cudaGetLastError() );

	}

	printf( "Copying result...\n" );
	for( int i = 0; i < deviceCount; i++ )
	{
		int device = usedDevices[ i ];
		CUDA_CHECK_RETURN( cudaSetDevice( device ) );

		//
		int arrayOffset = pointsChunksPerDevice * BLOCK_DIM * i * n;
		CALC_POINT_NUMBER_PER_DEVICE

		int bytesCount = pointsPerCurrentDevice * n * sizeof( FP );
		printf( "Copying result from GPU%d, %d bytes\n", device, bytesCount );
		CUDA_CHECK_RETURN( cudaMemcpy( points + arrayOffset, d_points[ i ], bytesCount, cudaMemcpyDeviceToHost ) );
		
	}


	// ???
	uint32_t funcSize = func.size();
	func.clear();
	printf( "Storing result...\n" );	
	for( uint32_t i = 0; i < funcSize; i++ )
	{
		FPVector x( dimX );
		uint32_t offsetToPoint = ( i - i % BLOCK_DIM ) * n + ( i % BLOCK_DIM );

		for( uint32_t j = 0; j < dimX; j++ )
			x[ j ] = points[ offsetToPoint + j * BLOCK_DIM ];

		func.define( x ) = points[ offsetToPoint + ( n - 1 ) * BLOCK_DIM ];
	}

	for( int i = 0; i < deviceCount; i++ )
	{
		int device = usedDevices[ i ];
		CUDA_CHECK_RETURN( cudaSetDevice( device ) ); 
		CUDA_CHECK_RETURN( cudaFree( ( void* )d_hyperplanes[ i ] ) );
		CUDA_CHECK_RETURN( cudaFree( ( void* )d_points[ i ] ) );
		//
		CUDA_CHECK_RETURN( cudaDeviceReset() );
		CUDA_CHECK_RETURN( cudaGetLastError() );
	}

	delete[] hyperplanes;
	delete[] points;

	printf( "Done\n" );
}

}