#include "ScalarFunction.hpp"

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


//
const int BLOCK_DIM = 192;
const int MAX_THREADS_PER_DEVICE = 1536;


//
__global__ void kernel1( FP* hyperplanes, FP* points, int32_t n, int32_t dimX, int32_t numberOfHyperplanes, int32_t numberOfPoints )
{
	if( blockIdx.x * blockDim.x + threadIdx.x >= numberOfHyperplanes )
		return;

	int32_t offsetToHyperplanesChunk = blockIdx.x * blockDim.x * ( n + 1 );

	FP resultDistance = 0.0;
	for( int32_t k = 0; k < numberOfPoints * n; k += n * BLOCK_DIM )
		#pragma unroll
		for( int32_t i = 0; i < BLOCK_DIM; i++ )
		{
			FP d = 0.0;

			int32_t offsetToPoint = k + i;

			// dot product of point and normal is distance
			int16_t j = 0;
			for( ; j < dimX * BLOCK_DIM; j += BLOCK_DIM ) 
				d += points[ offsetToPoint + j ] * hyperplanes[ offsetToHyperplanesChunk + threadIdx.x + j ];
			d += points[ offsetToPoint + j ] * hyperplanes[ offsetToHyperplanesChunk + threadIdx.x + j ]; 

			if( d > resultDistance )
				resultDistance = d;
		}
 
    hyperplanes[ offsetToHyperplanesChunk + threadIdx.x + n * BLOCK_DIM ] = resultDistance;
}


//
__global__ void kernel1_1( FP** hyperplanes, int32_t deviceCount, int32_t n, int32_t numberOfHyperplanes )
{
	if( blockIdx.x * blockDim.x + threadIdx.x >= numberOfHyperplanes )
		return;

	int32_t offset = blockIdx.x * blockDim.x * ( n + 1 ) + threadIdx.x + n * BLOCK_DIM;

	FP resultDistance = hyperplanes[ 0 ][ offset ];
	for( int32_t i = 1; i < deviceCount; i++ )
	{
		if( hyperplanes[ i ][ offset ] > resultDistance )
			resultDistance = hyperplanes[ i ][ offset ];
	}
	hyperplanes[ 0 ][ offset ] = resultDistance;
}


// sending dimX as argument is to reduce registers usage
__global__ void kernel2( FP* hyperplanes, FP* points, int32_t n, int dimX, int32_t numberOfHyperplanes, int32_t numberOfPoints )
{
	if( blockIdx.x * blockDim.x + gridDim.x * blockDim.x * blockIdx.y + threadIdx.x >= numberOfPoints )
		return;

	int32_t offsetToPointsChunk = ( blockIdx.x * blockDim.x + gridDim.x * blockDim.x * blockIdx.y ) * n;

	FP funcVal = points[ offsetToPointsChunk + threadIdx.x + ( n - 1 ) * BLOCK_DIM ];
	FP convexVal = funcVal;

	for( int32_t i = 0; i < numberOfHyperplanes; i++ )
	{
		int32_t offsetToHyperplane = ( i - i % BLOCK_DIM ) * ( n + 1 ) + ( i % BLOCK_DIM );

		FP val = 0.0;
		// xi - iter->first
		// Ni - hyperplane normal
		// val = x(n - 1) = ( -N0*x0 - N1*x1 - ... - N(n - 2)*x(n - 2) + xn ) / N(n - 1)
		int16_t j = 0;
		for( ; j < dimX * BLOCK_DIM; j += BLOCK_DIM )
			val -= points[ offsetToPointsChunk + threadIdx.x + j ] * hyperplanes[ offsetToHyperplane + j ];
		val += hyperplanes[ offsetToHyperplane + j + BLOCK_DIM ];
		val /= hyperplanes[ offsetToHyperplane + j ] + EPSILON;

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
void ScalarFunction::CopyData( const int& dimX )
{
	const int n = dimX + 1;

	int i = 0;
	for( ScalarFunction::iterator iter = begin(); iter != end(); ++iter, i++ )
	{
		int offsetToPoint = ( i - i % BLOCK_DIM ) * n + ( i % BLOCK_DIM );

		for( int j = 0; j < dimX; j++ )
			points[ offsetToPoint + j * BLOCK_DIM ] = iter->first[ j ];

		points[ offsetToPoint + ( n - 1 ) * BLOCK_DIM ] = iter->second;
	}

	for( ; i < pointsArraySize / n; i++ )
	{
		int offsetToPoint = ( i - i % BLOCK_DIM ) * n + ( i % BLOCK_DIM );

		for( int j = 0; j < dimX; j++ )
			points[ offsetToPoint + j * BLOCK_DIM ] = 0.0;

		points[ offsetToPoint + ( n - 1 ) * BLOCK_DIM ] = 0.0;
	}
}


//
void ScalarFunction::InitHyperplanes( const int& dimX, const int& numberOfHyperplanes, const FP& dFi )
{
	FPVector fi( dimX, 0.0 );

	const int n = dimX + 1;

	for( int i = 0; i < numberOfHyperplanes; i++ )
	{
		int offset = ( i - i % BLOCK_DIM ) * ( n + 1 ) + ( i % BLOCK_DIM );

		for( int j = 0; j < n; j++ )
		{
			FP* normalComponent = &hyperplanes[ offset + j * BLOCK_DIM ];

			*normalComponent = 1.0;
			for( int k = 0; k < j; k++ )
				*normalComponent *= sin( fi[ k ] );

			if( j != n - 1 )
				*normalComponent *= cos( fi[ j ] );
		}

		// not good enough
		bool shift = true;
		for( int k = 0; ( k < dimX ) && shift; k++ )
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
}


//
int ScalarFunction::PrepareDevices( const int& neededDeviceNumber )
{
	int deviceCount;
	CUDA_CHECK_RETURN( cudaGetDeviceCount( &deviceCount ) );
	printf( "Available device count: %d, needed device count: %d\n", deviceCount, neededDeviceNumber );
	if( deviceCount > MAX_GPU_COUNT )
	{
		printf( "Too much GPUs %d\n", deviceCount );
		deviceCount = MAX_GPU_COUNT;
	}

	// 
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

	if( deviceCount > neededDeviceNumber )
	{
		if( neededDeviceNumber <= ( int )devicesGroups[ 0 ].size() )
		{
			int devicesToRemove = ( int )devicesGroups[ 0 ].size() - neededDeviceNumber;
			devicesGroups[ 0 ].erase( devicesGroups[ 0 ].end() - devicesToRemove, devicesGroups[ 0 ].end() );
			devicesGroups[ 1 ].clear();
		}
		else
		{
			int devicesToRemove = deviceCount - neededDeviceNumber;
			devicesGroups[ 1 ].erase( devicesGroups[ 1 ].end() - devicesToRemove, devicesGroups[ 1 ].end() );
		}
		deviceCount = neededDeviceNumber;
	}

	// enabling peer access
	for( int j = 0; j < 2; j++ )
	{
		if( devicesGroups[ j ].size() == 0 )
			continue;

		CUDA_CHECK_RETURN( cudaSetDevice( devicesGroups[ j ][ 0 ] ) );
		for( int i = 1; i < ( int )devicesGroups[ j ].size(); i++ )	
		{
			CUDA_CHECK_RETURN( cudaDeviceEnablePeerAccess( devicesGroups[ j ][ i ], 0 ) );
		}
	}

	printf( "Used devices in gpoup 1: %u, group 2: %u\n", devicesGroups[ 0 ].size(), devicesGroups[ 1 ].size() );

	return deviceCount;
}


//
void ScalarFunction::DeviceMemoryPreparing( const int& n, const int& deviceCount )
{
	printf( "Memory preparing...\n" );
	for( int j = 0; j < 2; j++ )
		for( int i = 0; i < ( int )devicesGroups[ j ].size(); i++ )
		{
			int device = devicesGroups[ j ][ i ];
			CUDA_CHECK_RETURN( cudaSetDevice( device ) );
			// optimization
			CUDA_CHECK_RETURN( cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 ) );

			//
			CUDA_CHECK_RETURN( cudaMalloc( &d_hyperplanes[ device ], hyperplanesArraySize * sizeof( FP ) ) );

			if( i == 0 )
			{
				CUDA_CHECK_RETURN( cudaMemcpy( d_hyperplanes[ device ], hyperplanes, hyperplanesArraySize * sizeof( FP ), cudaMemcpyHostToDevice ) );
			}
			else
			{
				// TODO: smart copying, pair
				int lastDevice = devicesGroups[ j ][ i - 1 ];
				CUDA_CHECK_RETURN( cudaMemcpyPeer( d_hyperplanes[ device ], device, d_hyperplanes[ lastDevice ], lastDevice, hyperplanesArraySize * sizeof( FP ) ) );
			}

			int arrayOffset = pointsChunksPerDevice * BLOCK_DIM * device * n;

			//
			int bytesCount = CalcPointsNumberPerDevice( device, deviceCount ) * n * sizeof( FP );
			CUDA_CHECK_RETURN( cudaMalloc( &d_points[ device ], bytesCount ) );
			CUDA_CHECK_RETURN( cudaMemcpy( d_points[ device ], points + arrayOffset, bytesCount, cudaMemcpyHostToDevice ) );
		}
}


//
int ScalarFunction::CalcPointsNumberPerDevice( const int& device, const int& deviceCount )
{
	return ( ( device == deviceCount - 1 ) ? pointsChunksForLastDevice : pointsChunksPerDevice ) * BLOCK_DIM;
}


//
void ScalarFunction::Synchronize()
{
	printf( "Synchronizing...\n" );
	for( int j = 0; j < 2; j++ )
		for( int i = 0; i < ( int )devicesGroups[ j ].size(); i++ )
		{
			int device = devicesGroups[ j ][ i ];
			CUDA_CHECK_RETURN( cudaSetDevice( device ) );
			CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
			CUDA_CHECK_RETURN( cudaGetLastError() );
		}
}


//
void ScalarFunction::FirstStage( const int& dimX, const int& numberOfHyperplanes, const int& deviceCount )
{
	const int n = dimX + 1;
	dim3 gridDim, blockDim;

	printf( "Running first kernel...\n" );
	for( int j = 0; j < 2; j++ )
		for( int i = 0; i < ( int )devicesGroups[ j ].size(); i++ )
		{
			int device = devicesGroups[ j ][ i ]; 
			CUDA_CHECK_RETURN( cudaSetDevice( device ) );	

			getGridAndBlockDim( numberOfHyperplanes, gridDim, blockDim, device );
			kernel1<<< gridDim, blockDim >>>( d_hyperplanes[ device ], d_points[ device ], n, dimX, numberOfHyperplanes, CalcPointsNumberPerDevice( device, deviceCount ) );
		}

	Synchronize();
}


//
void ScalarFunction::SecondStage( const int& dimX, const int& numberOfHyperplanes )
{
	const int n = dimX + 1;
	dim3 gridDim, blockDim;

	for( int j = 0; j < 2; j++ )
	{
		// no need to run next kernel if device only one
		int deviceCount = ( int )devicesGroups[ j ].size();
		if( deviceCount > 1 )
		{
			//
			printf( "Running second kernel...\n" );

			int device = devicesGroups[ j ][ 0 ];
			CUDA_CHECK_RETURN( cudaSetDevice( device ) );

			FP** hostAllocatedMem;
			cudaHostAlloc( ( void** )&hostAllocatedMem, deviceCount * sizeof( FP* ), cudaHostAllocDefault );
			for( int i = 0; i < deviceCount; i++ )
				hostAllocatedMem[ i ] = d_hyperplanes[ devicesGroups[ j ][ i ] ];

			getGridAndBlockDim( numberOfHyperplanes, gridDim, blockDim, device );
			kernel1_1<<< gridDim, blockDim >>>( hostAllocatedMem, deviceCount, n, numberOfHyperplanes );

			printf( "Synchronizing...\n" );
			CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
			CUDA_CHECK_RETURN( cudaGetLastError() );
			CUDA_CHECK_RETURN( cudaFreeHost( hostAllocatedMem ) );
		}
	}

	if( devicesGroups[ 1 ].size() > 0 )
	{
		printf( "Running second kernel...\n" );
		int device = devicesGroups[ 0 ][ 0 ];
		int deviceCount = 2;
		CUDA_CHECK_RETURN( cudaSetDevice( device ) );		

		FP** hostAllocatedMem;
		cudaHostAlloc( ( void** )&hostAllocatedMem, deviceCount * sizeof( FP* ), cudaHostAllocDefault );
		for( int i = 0; i < deviceCount; i++ )
			hostAllocatedMem[ i ] = d_hyperplanes[ devicesGroups[ 0 ][ i ] ];

		int srcDevice = devicesGroups[ 1 ][ 0 ];
		CUDA_CHECK_RETURN( cudaMemcpyPeer( hostAllocatedMem[ 1 ], devicesGroups[ 0 ][ 1 ], d_hyperplanes[ srcDevice ], srcDevice, hyperplanesArraySize * sizeof( FP ) ) );

		getGridAndBlockDim( numberOfHyperplanes, gridDim, blockDim, device );
		kernel1_1<<< gridDim, blockDim >>>( hostAllocatedMem, deviceCount, n, numberOfHyperplanes );

		CUDA_CHECK_RETURN( cudaMemcpyPeer( d_hyperplanes[ srcDevice ], srcDevice, hostAllocatedMem[ 0 ], devicesGroups[ 0 ][ 0 ], hyperplanesArraySize * sizeof( FP ) ) );

		printf( "Synchronizing...\n" );
		CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
		CUDA_CHECK_RETURN( cudaGetLastError() );
		CUDA_CHECK_RETURN( cudaFreeHost( hostAllocatedMem ) );
	}
}


//
void ScalarFunction::ThirdStage( const int& dimX, const int& numberOfHyperplanes, const int& deviceCount )
{
	const int n = dimX + 1;
	dim3 gridDim, blockDim;

	printf( "Running third kernel...\n" );
	for( int j = 0; j < 2; j++ )
		for( int i = 0; i < ( int )devicesGroups[ j ].size(); i++ )
		{
			int device = devicesGroups[ j ][ i ];
			CUDA_CHECK_RETURN( cudaSetDevice( device ) );

			// copy hyperplanes from first device to others
			if( i != 0 )
			{
				int lastDevice = devicesGroups[ j ][ i - 1 ];
				CUDA_CHECK_RETURN( cudaMemcpyPeer( d_hyperplanes[ device ], device, d_hyperplanes[ lastDevice ], lastDevice, hyperplanesArraySize * sizeof( FP ) ) );
			}

			int pointsPerCurrentDevice = CalcPointsNumberPerDevice( device, deviceCount );

			getGridAndBlockDim( pointsPerCurrentDevice, gridDim, blockDim, device );
			kernel2<<< gridDim, blockDim >>>( d_hyperplanes[ device ], d_points[ device ], n, dimX, numberOfHyperplanes, pointsPerCurrentDevice );
		}

	//
	Synchronize();
}


//
void ScalarFunction::GetResult( const int& dimX, const int& deviceCount )
{
	const int n = dimX + 1;

	printf( "Copying result...\n" );
	for( int j = 0; j < 2; j++ )
		for( int i = 0; i < ( int )devicesGroups[ j ].size(); i++ )
		{
			int device = devicesGroups[ j ][ i ];
			CUDA_CHECK_RETURN( cudaSetDevice( device ) );

			//
			int arrayOffset = pointsChunksPerDevice * BLOCK_DIM * device * n;

			int bytesCount = CalcPointsNumberPerDevice( device, deviceCount ) * n * sizeof( FP );
			printf( "Copying result from GPU%d, %d bytes\n", device, bytesCount );
			CUDA_CHECK_RETURN( cudaMemcpy( points + arrayOffset, d_points[ device ], bytesCount, cudaMemcpyDeviceToHost ) );			
		}


	// ???
	uint32_t funcSize = size();
	clear();
	printf( "Storing result...\n" );	
	FPVector x( dimX );
	for( uint32_t i = 0; i < funcSize; i++ )
	{
		uint32_t offsetToPoint = ( i - i % BLOCK_DIM ) * n + ( i % BLOCK_DIM );

		for( uint32_t j = 0; j < dimX; j++ )
			x[ j ] = points[ offsetToPoint + j * BLOCK_DIM ];

		define( x ) = points[ offsetToPoint + ( n - 1 ) * BLOCK_DIM ];
	}
}


//
void ScalarFunction::makeConvexGPU( const int& dimX, const int& numberOfPoints )
{
	if( dimX == 0 )
		return;
	
	FP dFi = PI / ( numberOfPoints - 1 );

	int n = dimX + 1; // space dimension

	int numberOfHyperplanes = pow( numberOfPoints, dimX );

	// first x0.. x(n - 2) elements are independent vars. in 2D it will be x
	// x(n - 1) element dependent var. . in 2D it will be y
	// xn - constant, represents distance between O and hyperplane
	hyperplanesArraySize = ( numberOfHyperplanes + ( ( numberOfHyperplanes % BLOCK_DIM == 0 ) ? 0 : BLOCK_DIM - numberOfHyperplanes % BLOCK_DIM ) ) * ( n + 1 );
	hyperplanes = new FP[ hyperplanesArraySize ];

	int pointsNum = ( size() + ( ( size() % BLOCK_DIM == 0 ) ? 0 : BLOCK_DIM - size() % BLOCK_DIM ) );
	pointsArraySize = pointsNum * n;
	points = new FP[ pointsArraySize ];

	printf( "Number of hyperplanes: %d\n", numberOfHyperplanes );
	printf( "Number of points: %d, unused: %d\n", pointsNum, ( int )( pointsNum - size() ) );

	printf( "Memory allocated for hyperplanes: %d\n", hyperplanesArraySize * sizeof( FP ) );
	printf( "Memory allocated for points: %d\n", pointsArraySize * sizeof( FP ) );

	CopyData( dimX );

	InitHyperplanes( dimX, numberOfHyperplanes, dFi );

	int neededDeviceNumber = pointsNum / MAX_THREADS_PER_DEVICE;
	if( neededDeviceNumber == 0 ) neededDeviceNumber = 1;
	neededDeviceNumber = neededDeviceNumber > MAX_GPU_COUNT ? MAX_GPU_COUNT : neededDeviceNumber;

	int deviceCount = PrepareDevices( neededDeviceNumber );

	// общее кол-во чанков из точек
	pointsChunksNumber = pointsNum / BLOCK_DIM;
	// чанков на одну видеокарту
	pointsChunksPerDevice = pointsChunksNumber / deviceCount;
	// чанков на последнюю видеокарту
	pointsChunksForLastDevice = pointsChunksNumber - pointsChunksPerDevice * ( deviceCount - 1 );

	//
	DeviceMemoryPreparing( n, deviceCount );

	//
	FirstStage( dimX, numberOfHyperplanes, deviceCount );
	SecondStage( dimX, numberOfHyperplanes );
	ThirdStage( dimX, numberOfHyperplanes, deviceCount );
	GetResult( dimX, deviceCount );

	//
	for( int j = 0; j < 2; j++ )
		for( int i = 0; i < ( int )devicesGroups[ j ].size(); i++ )
		{
			int device = devicesGroups[ j ][ i ];
			CUDA_CHECK_RETURN( cudaSetDevice( device ) ); 
			CUDA_CHECK_RETURN( cudaFree( ( void* )d_hyperplanes[ device ] ) );
			CUDA_CHECK_RETURN( cudaFree( ( void* )d_points[ device ] ) );
			//
			CUDA_CHECK_RETURN( cudaDeviceReset() );
			CUDA_CHECK_RETURN( cudaGetLastError() );
		}

	delete[] hyperplanes;
	delete[] points;

	printf( "Done\n" );
}