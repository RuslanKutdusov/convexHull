#include "ScalarFunction.hpp"

#include <stdio.h>
#include <stdint.h>
#include <vector>
#include <boost/static_assert.hpp>

//
#define CUDA_CHECK_RETURN( value ) {											\
	cudaError_t _m_cudaStat = value;										\
	if ( _m_cudaStat != cudaSuccess ) {										\
		fprintf( stderr, "Error '%s' at line %d in file %s\n",					\
				cudaGetErrorString( _m_cudaStat ), __LINE__, __FILE__ );		\
		exit( 1 );															\
	} }


//
const uint32_t BLOCK_DIM = 192;
const uint32_t MAX_THREADS_PER_DEVICE = 1536;


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
__global__ void kernel1_1( FP** hyperplanes, uint32_t deviceCount, uint32_t n, uint32_t numberOfHyperplanes )
{
	if( blockIdx.x * blockDim.x + threadIdx.x >= numberOfHyperplanes )
		return;

	uint32_t offset = blockIdx.x * blockDim.x * ( n + 1 ) + threadIdx.x + n * BLOCK_DIM;

	FP resultDistance = hyperplanes[ 0 ][ offset ];
	for( uint32_t i = 1; i < deviceCount; i++ )
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
		uint16_t j = 0;
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
void getGridAndBlockDim( uint32_t n, dim3& gridDim, dim3& blockDim, uint32_t device )
{
	// gpu hardware limits for compute caps 1.2 and 2.0
	const uint32_t warpSize = 32;
	const uint32_t blocksPerSM = 8;

	cudaDeviceProp deviceProp;
	CUDA_CHECK_RETURN( cudaGetDeviceProperties( &deviceProp, device ) );

	uint32_t warpCount = ( n / warpSize ) + ( ( ( n % warpSize ) == 0 ) ? 0 : 1 );

	uint32_t threadsPerBlock = deviceProp.maxThreadsPerMultiProcessor / blocksPerSM;
	uint32_t warpsPerBlock = threadsPerBlock / warpSize;

	uint32_t blockCount = ( warpCount / warpsPerBlock ) + ( ( ( warpCount % warpsPerBlock ) == 0 ) ? 0 : 1 );

	blockDim = dim3( threadsPerBlock, 1, 1 );

	gridDim = dim3( blockCount, 1, 1 );

	if( blockCount > deviceProp.maxGridSize[ 0 ] )
	{
		gridDim.x = gridDim.y = sqrt( blockCount );
		if( gridDim.x * gridDim.x < blockCount )
			gridDim.x += 1;
	}

	printf( "GPU%d: %s, Task size: %u, warp number: %u, threads per block: %u, warps per block: %u, grid: (%d, %d, 1)\n", device, deviceProp.name, n, warpCount, threadsPerBlock, warpsPerBlock, gridDim.x, gridDim.y );
}


//
void ScalarFunction::CopyData( const uint32_t& dimX )
{
	const uint32_t n = dimX + 1;

	uint32_t i = 0;
	for( ScalarFunction::iterator iter = begin(); iter != end(); ++iter, i++ )
	{
		uint32_t offsetToPoint = ( i - i % BLOCK_DIM ) * n + ( i % BLOCK_DIM );

		for( uint32_t j = 0; j < dimX; j++ )
			points[ offsetToPoint + j * BLOCK_DIM ] = iter->first[ j ];

		points[ offsetToPoint + ( n - 1 ) * BLOCK_DIM ] = iter->second;
	}

	for( ; i < pointsArraySize / n; i++ )
	{
		uint32_t offsetToPoint = ( i - i % BLOCK_DIM ) * n + ( i % BLOCK_DIM );

		for( uint32_t j = 0; j < dimX; j++ )
			points[ offsetToPoint + j * BLOCK_DIM ] = 0.0;

		points[ offsetToPoint + ( n - 1 ) * BLOCK_DIM ] = 0.0;
	}
}


//
void ScalarFunction::InitHyperplanes( const uint32_t& dimX, const uint32_t& numberOfHyperplanes, const FP& dFi )
{
	FPVector fi( dimX, 0.0 );

	const uint32_t n = dimX + 1;

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
}


//
uint32_t ScalarFunction::PrepareDevices( const uint32_t& neededDeviceNumber )
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
	for( uint32_t j = 1; j < deviceCount; j++ )
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
		if( neededDeviceNumber <= devicesGroups[ 0 ].size() )
		{
			uint32_t devicesToRemove = devicesGroups[ 0 ].size() - neededDeviceNumber;
			devicesGroups[ 0 ].erase( devicesGroups[ 0 ].end() - devicesToRemove, devicesGroups[ 0 ].end() );
			devicesGroups[ 1 ].clear();
		}
		else
		{
			uint32_t devicesToRemove = deviceCount - neededDeviceNumber;
			devicesGroups[ 1 ].erase( devicesGroups[ 1 ].end() - devicesToRemove, devicesGroups[ 1 ].end() );
		}
		deviceCount = neededDeviceNumber;
	}

	// enabling peer access
	for( uint32_t j = 0; j < 2; j++ )
	{
		if( devicesGroups[ j ].size() == 0 )
			continue;

		for( uint32_t i = 0; i < devicesGroups[ j ].size(); i++ )	
		{
			CUDA_CHECK_RETURN( cudaSetDevice( devicesGroups[ j ][ i ] ) );
			for( uint32_t k = 0; k < devicesGroups[ j ].size(); k++ )
			{
				if( i == k )
					continue;

				CUDA_CHECK_RETURN( cudaDeviceEnablePeerAccess( devicesGroups[ j ][ k ], 0 ) );
			}
		}
	}

	for( uint32_t j = 0; j < 2; j++ )
		for( uint32_t i = 0; i < devicesGroups[ j ].size(); i++ )
		{
			uint32_t device = devicesGroups[ j ][ i ];
			CUDA_CHECK_RETURN( cudaSetDevice( device ) );
			CUDA_CHECK_RETURN( cudaEventCreate( ( cudaEvent_t* )&start[ device ] ) );
    		CUDA_CHECK_RETURN( cudaEventCreate( ( cudaEvent_t* )&stop[ device ] ) );
    	}

	printf( "Used devices in gpoup 1: %u, group 2: %u\n", devicesGroups[ 0 ].size(), devicesGroups[ 1 ].size() );

	return deviceCount;
}


//
void ScalarFunction::DeviceMemoryPreparing( const uint32_t& n, const uint32_t& deviceCount )
{
	printf( "Memory preparing...\n" );
	for( uint32_t j = 0; j < 2; j++ )
		for( uint32_t i = 0; i < devicesGroups[ j ].size(); i++ )
		{
			uint32_t device = devicesGroups[ j ][ i ];
			CUDA_CHECK_RETURN( cudaSetDevice( device ) );
			// optimization
			CUDA_CHECK_RETURN( cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 ) );

			//
			CUDA_CHECK_RETURN( cudaEventRecord( ( cudaEvent_t )start[ device ], 0 ) );

			//
			CUDA_CHECK_RETURN( cudaMalloc( &d_hyperplanes[ device ], hyperplanesArraySize * sizeof( FP ) ) );

			if( i == 0 )
			{
				CUDA_CHECK_RETURN( cudaMemcpy( d_hyperplanes[ device ], hyperplanes, hyperplanesArraySize * sizeof( FP ), cudaMemcpyHostToDevice ) );
			}
			else
			{
				// TODO: smart copying, pair
				uint32_t lastDevice = devicesGroups[ j ][ i - 1 ];
				CUDA_CHECK_RETURN( cudaMemcpyPeer( d_hyperplanes[ device ], device, d_hyperplanes[ lastDevice ], lastDevice, hyperplanesArraySize * sizeof( FP ) ) );
			}

			uint32_t arrayOffset = pointsChunksPerDevice * BLOCK_DIM * device * n;

			//
			uint32_t bytesCount = CalcPointsNumberPerDevice( device, deviceCount ) * n * sizeof( FP );
			CUDA_CHECK_RETURN( cudaMalloc( &d_points[ device ], bytesCount ) );
			CUDA_CHECK_RETURN( cudaMemcpy( d_points[ device ], points + arrayOffset, bytesCount, cudaMemcpyHostToDevice ) );

			//
			CUDA_CHECK_RETURN( cudaEventRecord( ( cudaEvent_t )stop[ device ], 0 ) );
		}

	Synchronize( LAUNCH_TIME_HTOD );
}


//
uint32_t ScalarFunction::CalcPointsNumberPerDevice( const uint32_t& device, const uint32_t& deviceCount )
{
	return ( ( device == deviceCount - 1 ) ? pointsChunksForLastDevice : pointsChunksPerDevice ) * BLOCK_DIM;
}


//
void ScalarFunction::Synchronize( LAUNCH_TIME lt )
{
	printf( "Synchronizing...\n" );
	for( uint32_t j = 0; j < 2; j++ )
		for( uint32_t i = 0; i < devicesGroups[ j ].size(); i++ )
		{
			uint32_t device = devicesGroups[ j ][ i ];
			CUDA_CHECK_RETURN( cudaSetDevice( device ) );
			CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
			CUDA_CHECK_RETURN( cudaGetLastError() );

			FixLaunchTime( lt, device );
		}
}


//
void ScalarFunction::FixLaunchTime( LAUNCH_TIME lt, uint32_t device )
{
	CUDA_CHECK_RETURN( cudaEventElapsedTime( &launchTime[ lt ][ device ], ( cudaEvent_t )start[ device ], ( cudaEvent_t )stop[ device ] ) );
}


//
void ScalarFunction::FirstStage( const uint32_t& dimX, const uint32_t& numberOfHyperplanes, const uint32_t& deviceCount )
{
	const uint32_t n = dimX + 1;
	dim3 gridDim, blockDim;

	printf( "Running first kernel...\n" );
	for( uint32_t j = 0; j < 2; j++ )
		for( uint32_t i = 0; i < devicesGroups[ j ].size(); i++ )
		{
			uint32_t device = devicesGroups[ j ][ i ]; 
			CUDA_CHECK_RETURN( cudaSetDevice( device ) );	

			getGridAndBlockDim( numberOfHyperplanes, gridDim, blockDim, device );
			CUDA_CHECK_RETURN( cudaEventRecord( ( cudaEvent_t )start[ device ], 0 ) );
			kernel1<<< gridDim, blockDim >>>( d_hyperplanes[ device ], d_points[ device ], n, dimX, numberOfHyperplanes, CalcPointsNumberPerDevice( device, deviceCount ) );
			CUDA_CHECK_RETURN( cudaEventRecord( ( cudaEvent_t )stop[ device ], 0 ) );
		}

	Synchronize( LAUNCH_TIME_STAGE1 );
}


//
void ScalarFunction::SecondStage( const uint32_t& dimX, const uint32_t& numberOfHyperplanes )
{
	const uint32_t n = dimX + 1;
	dim3 gridDim, blockDim;

	for( uint32_t j = 0; j < 2; j++ )
	{
		// no need to run next kernel if device only one
		uint32_t deviceCount = devicesGroups[ j ].size();
		if( deviceCount > 1 )
		{
			//
			printf( "Running second kernel...\n" );

			uint32_t device = devicesGroups[ j ][ 0 ];
			CUDA_CHECK_RETURN( cudaSetDevice( device ) );

			FP** hostAllocatedMem;
			cudaHostAlloc( ( void** )&hostAllocatedMem, deviceCount * sizeof( FP* ), cudaHostAllocDefault );
			for( uint32_t i = 0; i < deviceCount; i++ )
				hostAllocatedMem[ i ] = d_hyperplanes[ devicesGroups[ j ][ i ] ];

			getGridAndBlockDim( numberOfHyperplanes, gridDim, blockDim, device );
			CUDA_CHECK_RETURN( cudaEventRecord( ( cudaEvent_t )start[ device ], 0 ) );
			kernel1_1<<< gridDim, blockDim >>>( hostAllocatedMem, deviceCount, n, numberOfHyperplanes );
			CUDA_CHECK_RETURN( cudaEventRecord( ( cudaEvent_t )stop[ device ], 0 ) );

			printf( "Synchronizing...\n" );
			CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
			CUDA_CHECK_RETURN( cudaGetLastError() );
			CUDA_CHECK_RETURN( cudaFreeHost( hostAllocatedMem ) );

			j ? FixLaunchTime( LAUNCH_TIME_STAGE2_FIRST_GROUP, device ) : FixLaunchTime( LAUNCH_TIME_STAGE2_SECOND_GROUP, device );
		}
	}

	if( devicesGroups[ 1 ].size() > 0 )
	{
		printf( "Running second kernel...\n" );
		uint32_t device = devicesGroups[ 0 ][ 0 ];
		uint32_t deviceCount = 2;
		CUDA_CHECK_RETURN( cudaSetDevice( device ) );		

		FP** hostAllocatedMem;
		cudaHostAlloc( ( void** )&hostAllocatedMem, deviceCount * sizeof( FP* ), cudaHostAllocDefault );
		for( uint32_t i = 0; i < deviceCount; i++ )
			hostAllocatedMem[ i ] = d_hyperplanes[ devicesGroups[ 0 ][ i ] ];

		uint32_t srcDevice = devicesGroups[ 1 ][ 0 ];
		CUDA_CHECK_RETURN( cudaMemcpyPeer( hostAllocatedMem[ 1 ], devicesGroups[ 0 ][ 1 ], d_hyperplanes[ srcDevice ], srcDevice, hyperplanesArraySize * sizeof( FP ) ) );

		getGridAndBlockDim( numberOfHyperplanes, gridDim, blockDim, device );
		CUDA_CHECK_RETURN( cudaEventRecord( ( cudaEvent_t )start[ device ], 0 ) );
		kernel1_1<<< gridDim, blockDim >>>( hostAllocatedMem, deviceCount, n, numberOfHyperplanes );
		CUDA_CHECK_RETURN( cudaEventRecord( ( cudaEvent_t )stop[ device ], 0 ) );

		CUDA_CHECK_RETURN( cudaMemcpyPeer( d_hyperplanes[ srcDevice ], srcDevice, hostAllocatedMem[ 0 ], devicesGroups[ 0 ][ 0 ], hyperplanesArraySize * sizeof( FP ) ) );

		printf( "Synchronizing...\n" );
		CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
		CUDA_CHECK_RETURN( cudaGetLastError() );
		CUDA_CHECK_RETURN( cudaFreeHost( hostAllocatedMem ) );

		FixLaunchTime( LAUNCH_TIME_STAGE2, device );
	}
}


//
void ScalarFunction::ThirdStage( const uint32_t& dimX, const uint32_t& numberOfHyperplanes, const uint32_t& deviceCount )
{
	const uint32_t n = dimX + 1;
	dim3 gridDim, blockDim;

	printf( "Running third kernel...\n" );
	// циклы объединять не стоит, иначе kernel-ы будут запускаться последовательно
	for( uint32_t j = 0; j < 2; j++ )
		for( uint32_t i = 0; i < devicesGroups[ j ].size(); i++ )
		{
			uint32_t device = devicesGroups[ j ][ i ];
			CUDA_CHECK_RETURN( cudaSetDevice( device ) );

			// copy hyperplanes from first device to others
			if( i != 0 )
			{
				uint32_t lastDevice = devicesGroups[ j ][ i - 1 ];
				CUDA_CHECK_RETURN( cudaEventRecord( ( cudaEvent_t )start[ device ], 0 ) );
				CUDA_CHECK_RETURN( cudaMemcpyPeer( d_hyperplanes[ device ], device, d_hyperplanes[ lastDevice ], lastDevice, hyperplanesArraySize * sizeof( FP ) ) );
				CUDA_CHECK_RETURN( cudaEventRecord( ( cudaEvent_t )stop[ device ], 0 ) );
			}

		}

	Synchronize( LAUNCH_TIME_COPY_HYPERPLANES );

	for( uint32_t j = 0; j < 2; j++ )
		for( uint32_t i = 0; i < devicesGroups[ j ].size(); i++ )
		{
			uint32_t device = devicesGroups[ j ][ i ];
			CUDA_CHECK_RETURN( cudaSetDevice( device ) );

			uint32_t pointsPerCurrentDevice = CalcPointsNumberPerDevice( device, deviceCount );

			getGridAndBlockDim( pointsPerCurrentDevice, gridDim, blockDim, device );
			CUDA_CHECK_RETURN( cudaEventRecord( ( cudaEvent_t )start[ device ], 0 ) );
			kernel2<<< gridDim, blockDim >>>( d_hyperplanes[ device ], d_points[ device ], n, dimX, numberOfHyperplanes, pointsPerCurrentDevice );
			CUDA_CHECK_RETURN( cudaEventRecord( ( cudaEvent_t )stop[ device ], 0 ) );
		}

	//
	Synchronize( LAUNCH_TIME_STAGE3 );
}


//
void ScalarFunction::GetResult( const uint32_t& dimX, const uint32_t& deviceCount )
{
	const uint32_t n = dimX + 1;

	printf( "Copying result...\n" );
	for( uint32_t j = 0; j < 2; j++ )
		for( uint32_t i = 0; i < devicesGroups[ j ].size(); i++ )
		{
			uint32_t device = devicesGroups[ j ][ i ];
			CUDA_CHECK_RETURN( cudaSetDevice( device ) );

			//
			CUDA_CHECK_RETURN( cudaEventRecord( ( cudaEvent_t )start[ device ], 0 ) );

			//
			uint32_t arrayOffset = pointsChunksPerDevice * BLOCK_DIM * device * n;

			uint32_t bytesCount = CalcPointsNumberPerDevice( device, deviceCount ) * n * sizeof( FP );
			printf( "Copying result from GPU%d, %d bytes\n", device, bytesCount );
			CUDA_CHECK_RETURN( cudaMemcpy( points + arrayOffset, d_points[ device ], bytesCount, cudaMemcpyDeviceToHost ) );			

			//
			CUDA_CHECK_RETURN( cudaEventRecord( ( cudaEvent_t )stop[ device ], 0 ) );
		}

	Synchronize( LAUNCH_TIME_DTOH );

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
void ScalarFunction::makeConvexGPU( const uint32_t& dimX, const uint32_t& numberOfPoints )
{
	BOOST_STATIC_ASSERT( sizeof( cudaEvent_t ) == sizeof( start[ 0 ] ) );
	BOOST_STATIC_ASSERT( sizeof( cudaEvent_t ) == sizeof( stop[ 0 ] ) );

	//
	for( uint32_t i = 0; i < LAUNCH_TIME_COUNT; i++ )
		for( uint32_t j = 0; j < MAX_GPU_COUNT; j++ )
			launchTime[ i ][ j ] = 0.0f;

	//
	if( dimX == 0 || numberOfPoints == 0 )
		return;
	
	FP dFi = PI / ( numberOfPoints - 1 );

	uint32_t n = dimX + 1; // space dimension

	uint32_t numberOfHyperplanes = pow( numberOfPoints, dimX );

	// first x0.. x(n - 2) elements are independent vars. in 2D it will be x
	// x(n - 1) element dependent var. . in 2D it will be y
	// xn - constant, represents distance between O and hyperplane
	hyperplanesArraySize = ( numberOfHyperplanes + ( ( numberOfHyperplanes % BLOCK_DIM == 0 ) ? 0 : BLOCK_DIM - numberOfHyperplanes % BLOCK_DIM ) ) * ( n + 1 );
	hyperplanes = new FP[ hyperplanesArraySize ];

	uint32_t pointsNum = ( size() + ( ( size() % BLOCK_DIM == 0 ) ? 0 : BLOCK_DIM - size() % BLOCK_DIM ) );
	pointsArraySize = pointsNum * n;
	points = new FP[ pointsArraySize ];

	printf( "Number of hyperplanes: %u\n", numberOfHyperplanes );
	printf( "Number of points: %u, unused: %u\n", pointsNum, pointsNum - size() );

	printf( "Memory allocated for hyperplanes: %llu\n", hyperplanesArraySize * sizeof( FP ) );
	printf( "Memory allocated for points: %llu\n", pointsArraySize * sizeof( FP ) );

	CopyData( dimX );

	InitHyperplanes( dimX, numberOfHyperplanes, dFi );

	uint32_t neededDeviceNumber = pointsNum / MAX_THREADS_PER_DEVICE;
	if( neededDeviceNumber == 0 ) neededDeviceNumber = 1;
	neededDeviceNumber = neededDeviceNumber > MAX_GPU_COUNT ? MAX_GPU_COUNT : neededDeviceNumber;

	uint32_t deviceCount = PrepareDevices( neededDeviceNumber );

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
	for( uint32_t j = 0; j < 2; j++ )
		for( uint32_t i = 0; i < devicesGroups[ j ].size(); i++ )
		{
			uint32_t device = devicesGroups[ j ][ i ];
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

	const char* launchNames[ LAUNCH_TIME_COUNT ] = { "MemCopyHTOD     ", 
													 "Stage1          ", 
													 "Stage2 1st group", 
													 "Stage2 2nd group", 
													 "Stage2          ", 
													 "Copy hyperplanes",
													 "Stage3          ", 
													 "MemCopyDTOH     " };
	printf( "                 | " );
	for( uint32_t j = 0; j < deviceCount; j++ )
		printf( "Device%u      | ", j );
	printf("\n");

	for( int i = 0; i < LAUNCH_TIME_COUNT; i++ )
	{
		printf( "%s | ", launchNames[ i ] );
		for( uint32_t j = 0; j < deviceCount; j++ )
		{
			printf("%12f | ", launchTime[ i ][ j ]);
		}
		printf("\n");
	}
}