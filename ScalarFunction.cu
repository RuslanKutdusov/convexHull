#include "ScalarFunction.hpp"

#include <stdio.h>
#include <stdint.h>
#include <vector>
#include <boost/static_assert.hpp>
#include <math_constants.h>

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
__global__ void FirstStageKernel( FP* hyperplanes, FP* points, uint32_t n, uint32_t dimX, uint32_t numberOfHyperplanes, uint32_t numberOfPoints )
{
	if( blockIdx.x * blockDim.x + gridDim.x * blockDim.x * blockIdx.y + threadIdx.x >= numberOfHyperplanes )
		return;

	uint32_t offsetToHyperplanesChunk = ( blockIdx.x * blockDim.x + gridDim.x * blockDim.x * blockIdx.y ) * ( n + 1 );

	FP resultDistance = -CUDART_INF;
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
__global__ void SecondStageKernel( FP** hyperplanes, uint32_t deviceCount, uint32_t n, uint32_t dimX, uint32_t numberOfHyperplanes, bool makeDivisions )
{
	if( blockIdx.x * blockDim.x + gridDim.x * blockDim.x * blockIdx.y + threadIdx.x >= numberOfHyperplanes )
		return;

	uint32_t offset = ( blockIdx.x * blockDim.x + gridDim.x * blockDim.x * blockIdx.y ) * ( n + 1 ) + threadIdx.x;

	FP resultDistance = hyperplanes[ 0 ][ offset + n * BLOCK_DIM ];
	for( uint32_t i = 1; i < deviceCount; i++ )
	{
		if( hyperplanes[ i ][ offset + n * BLOCK_DIM ] > resultDistance )
			resultDistance = hyperplanes[ i ][ offset + n * BLOCK_DIM ];
	}

	if( makeDivisions )
	{
		FP Nn_1 = hyperplanes[ 0 ][ offset + dimX * BLOCK_DIM ] + EPSILON;
		uint16_t j = 0;
		for( ; j < dimX * BLOCK_DIM; j += BLOCK_DIM )
			hyperplanes[ 0 ][ offset + j ] = hyperplanes[ 0 ][ offset + j ] / Nn_1;
		hyperplanes[ 0 ][ offset + n * BLOCK_DIM ] = resultDistance / Nn_1;
	}
	else
		hyperplanes[ 0 ][ offset + n * BLOCK_DIM ] = resultDistance;	
}


// sending dimX as argument is to reduce registers usage
__global__ void ThirdStageKernel( FP* hyperplanes, FP* points, uint32_t n, uint32_t dimX, uint32_t numberOfHyperplanes, uint32_t numberOfPoints )
{
	if( blockIdx.x * blockDim.x + gridDim.x * blockDim.x * blockIdx.y + threadIdx.x >= numberOfPoints )
		return;

	uint32_t offsetToPointsChunk = ( blockIdx.x * blockDim.x + gridDim.x * blockDim.x * blockIdx.y ) * n;

	FP convexVal = MAX_VAL;
	FP funcVal = points[ offsetToPointsChunk + threadIdx.x + ( n - 1 ) * BLOCK_DIM ];

	for( uint32_t i = 0; i < numberOfHyperplanes; i++ )
	{
		uint32_t offsetToHyperplane = ( i - i % BLOCK_DIM ) * ( n + 1 ) + ( i % BLOCK_DIM );

		FP val = 0.0;
		// xi - iter->first
		// Ni - hyperplane normal
		// val = x(n - 1) = ( -N0*x0 - N1*x1 - ... - N(n - 2)*x(n - 2) + d ) / N(n - 1)
		uint16_t j = 0;
		for( ; j < dimX * BLOCK_DIM; j += BLOCK_DIM )
			val -= points[ offsetToPointsChunk + threadIdx.x + j ] * hyperplanes[ offsetToHyperplane + j ];
		val += hyperplanes[ offsetToHyperplane + j + BLOCK_DIM ];

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

	if( blockCount > ( uint32_t )deviceProp.maxGridSize[ 0 ] )
	{
		gridDim.x = gridDim.y = ( uint32_t )sqrtf( ( float )blockCount );
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
		uint64_t offsetToPoint = ( i - i % BLOCK_DIM ) * n + ( i % BLOCK_DIM );

		for( uint32_t j = 0; j < dimX; j++ )
			m_points[ offsetToPoint + j * BLOCK_DIM ] = iter->first[ j ];

		m_points[ offsetToPoint + ( n - 1 ) * BLOCK_DIM ] = iter->second;
	}

	for( ; i < m_pointsArraySize / n; i++ )
	{
		uint64_t offsetToPoint = ( i - i % BLOCK_DIM ) * n + ( i % BLOCK_DIM );

		for( uint32_t j = 0; j < dimX; j++ )
			m_points[ offsetToPoint + j * BLOCK_DIM ] = 0.0;

		m_points[ offsetToPoint + ( n - 1 ) * BLOCK_DIM ] = 0.0;
	}
}


//
void ScalarFunction::InitHyperplanes( const uint32_t& dimX, const uint32_t& numberOfHyperplanes, const FP& dFi )
{
	FPVector fi( dimX, 0.0 );

	const uint32_t n = dimX + 1;

	for( uint32_t i = 0; i < numberOfHyperplanes; i++ )
	{
		uint64_t offset = ( i - i % BLOCK_DIM ) * ( n + 1 ) + ( i % BLOCK_DIM );

		for( uint32_t j = 0; j < n; j++ )
		{
			FP* normalComponent = &m_hyperplanes[ offset + j * BLOCK_DIM ];

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
uint32_t ScalarFunction::PrepareDevices( const int32_t& neededDeviceNumber )
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
	m_devicesGroups[ 0 ].push_back( 0 );
	for( int32_t j = 1; j < deviceCount; j++ )
	{
		int accessible;
		cudaDeviceCanAccessPeer( &accessible, j, 0 );
		if( accessible )
			m_devicesGroups[ 0 ].push_back( j );
		else
			m_devicesGroups[ 1 ].push_back( j );
	}

	if( deviceCount > neededDeviceNumber )
	{
		if( neededDeviceNumber <= m_devicesGroups[ 0 ].size() )
		{
			uint32_t devicesToRemove = ( uint32_t )m_devicesGroups[ 0 ].size() - neededDeviceNumber;
			m_devicesGroups[ 0 ].erase( m_devicesGroups[ 0 ].end() - devicesToRemove, m_devicesGroups[ 0 ].end() );
			m_devicesGroups[ 1 ].clear();
		}
		else
		{
			uint32_t devicesToRemove = deviceCount - neededDeviceNumber;
			m_devicesGroups[ 1 ].erase( m_devicesGroups[ 1 ].end() - devicesToRemove, m_devicesGroups[ 1 ].end() );
		}
		deviceCount = neededDeviceNumber;
	}

	// enabling peer access
	for( uint32_t j = 0; j < 2; j++ )
	{
		if( m_devicesGroups[ j ].size() == 0 )
			continue;

		for( uint32_t i = 0; i < m_devicesGroups[ j ].size(); i++ )	
		{
			CUDA_CHECK_RETURN( cudaSetDevice( m_devicesGroups[ j ][ i ] ) );
			for( uint32_t k = 0; k < m_devicesGroups[ j ].size(); k++ )
			{
				if( i == k )
					continue;

				CUDA_CHECK_RETURN( cudaDeviceEnablePeerAccess( m_devicesGroups[ j ][ k ], 0 ) );
			}
		}
	}

	for( uint32_t j = 0; j < 2; j++ )
		for( uint32_t i = 0; i < m_devicesGroups[ j ].size(); i++ )
		{
			uint32_t device = m_devicesGroups[ j ][ i ];
			CUDA_CHECK_RETURN( cudaSetDevice( device ) );
			CUDA_CHECK_RETURN( cudaEventCreate( ( cudaEvent_t* )&m_start[ device ] ) );
    		CUDA_CHECK_RETURN( cudaEventCreate( ( cudaEvent_t* )&m_stop[ device ] ) );
    	}

	printf( "Used devices in gpoup 1: %u, group 2: %u\n", m_devicesGroups[ 0 ].size(), m_devicesGroups[ 1 ].size() );

	return deviceCount;
}


//
void ScalarFunction::DeviceMemoryPreparing( const uint32_t& n, const uint32_t& deviceCount )
{
	printf( "Memory preparing...\n" );
	for( uint32_t j = 0; j < 2; j++ )
		for( uint32_t i = 0; i < m_devicesGroups[ j ].size(); i++ )
		{
			uint32_t device = m_devicesGroups[ j ][ i ];
			CUDA_CHECK_RETURN( cudaSetDevice( device ) );
			// optimization
			CUDA_CHECK_RETURN( cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 ) );

			//
			CUDA_CHECK_RETURN( cudaEventRecord( ( cudaEvent_t )m_start[ device ], 0 ) );

			//
			CUDA_CHECK_RETURN( cudaMalloc( &m_hyperplanesDevPtr[ device ], m_hyperplanesArraySize * sizeof( FP ) ) );

			if( i == 0 )
			{
				CUDA_CHECK_RETURN( cudaMemcpy( m_hyperplanesDevPtr[ device ], m_hyperplanes, m_hyperplanesArraySize * sizeof( FP ), cudaMemcpyHostToDevice ) );
			}
			else
			{
				// TODO: smart copying, pair
				uint32_t lastDevice = m_devicesGroups[ j ][ i - 1 ];
				CUDA_CHECK_RETURN( cudaMemcpyPeer( m_hyperplanesDevPtr[ device ], device, m_hyperplanesDevPtr[ lastDevice ], lastDevice, m_hyperplanesArraySize * sizeof( FP ) ) );
			}

			uint64_t arrayOffset = m_pointsChunksPerDevice * BLOCK_DIM * device * n;

			//
			uint64_t bytesCount = CalcPointsNumberPerDevice( device, deviceCount ) * n * sizeof( FP );
			CUDA_CHECK_RETURN( cudaMalloc( &m_pointsDevPtr[ device ], bytesCount ) );
			CUDA_CHECK_RETURN( cudaMemcpy( m_pointsDevPtr[ device ], m_points + arrayOffset, bytesCount, cudaMemcpyHostToDevice ) );

			//
			CUDA_CHECK_RETURN( cudaEventRecord( ( cudaEvent_t )m_stop[ device ], 0 ) );
		}

	Synchronize( LAUNCH_TIME_HTOD );
}


//
uint32_t ScalarFunction::CalcPointsNumberPerDevice( const uint32_t& device, const uint32_t& deviceCount )
{
	return ( ( device == deviceCount - 1 ) ? m_pointsChunksForLastDevice : m_pointsChunksPerDevice ) * BLOCK_DIM;
}


//
void ScalarFunction::Synchronize( LAUNCH_TIME lt )
{
	printf( "Synchronizing...\n" );
	for( uint32_t j = 0; j < 2; j++ )
		for( uint32_t i = 0; i < m_devicesGroups[ j ].size(); i++ )
		{
			uint32_t device = m_devicesGroups[ j ][ i ];
			CUDA_CHECK_RETURN( cudaSetDevice( device ) );
			CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
			CUDA_CHECK_RETURN( cudaGetLastError() );

			FixLaunchTime( lt, device );
		}
}


//
void ScalarFunction::FixLaunchTime( LAUNCH_TIME lt, uint32_t device )
{
	CUDA_CHECK_RETURN( cudaEventElapsedTime( &m_launchTime[ lt ][ device ], ( cudaEvent_t )m_start[ device ], ( cudaEvent_t )m_stop[ device ] ) );
}


//
void ScalarFunction::FirstStage( const uint32_t& dimX, const uint32_t& numberOfHyperplanes, const uint32_t& deviceCount )
{
	const uint32_t n = dimX + 1;
	dim3 gridDim, blockDim;

	printf( "Running first stage kernel...\n" );
	for( uint32_t j = 0; j < 2; j++ )
		for( uint32_t i = 0; i < m_devicesGroups[ j ].size(); i++ )
		{
			uint32_t device = m_devicesGroups[ j ][ i ]; 
			CUDA_CHECK_RETURN( cudaSetDevice( device ) );	

			getGridAndBlockDim( numberOfHyperplanes, gridDim, blockDim, device );
			CUDA_CHECK_RETURN( cudaEventRecord( ( cudaEvent_t )m_start[ device ], 0 ) );
			FirstStageKernel<<< gridDim, blockDim >>>( m_hyperplanesDevPtr[ device ], m_pointsDevPtr[ device ], n, dimX, numberOfHyperplanes, CalcPointsNumberPerDevice( device, deviceCount ) );
			CUDA_CHECK_RETURN( cudaEventRecord( ( cudaEvent_t )m_stop[ device ], 0 ) );
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
		uint32_t deviceCount = ( uint32_t )m_devicesGroups[ j ].size();
		bool makeDivisions = m_devicesGroups[ 0 ].size() > 0 && m_devicesGroups[ 1 ].size() == 0 && j == 0;
		if( deviceCount > 1 || makeDivisions )
		{
			//
			printf( "Running second stage kernel...\n" );

			int32_t device = m_devicesGroups[ j ][ 0 ];
			CUDA_CHECK_RETURN( cudaSetDevice( device ) );

			FP** hostAllocatedMem;
			cudaHostAlloc( ( void** )&hostAllocatedMem, deviceCount * sizeof( FP* ), cudaHostAllocDefault );
			for( uint32_t i = 0; i < deviceCount; i++ )
				hostAllocatedMem[ i ] = m_hyperplanesDevPtr[ m_devicesGroups[ j ][ i ] ];

			getGridAndBlockDim( numberOfHyperplanes, gridDim, blockDim, device );
			CUDA_CHECK_RETURN( cudaEventRecord( ( cudaEvent_t )m_start[ device ], 0 ) );
			SecondStageKernel<<< gridDim, blockDim >>>( hostAllocatedMem, deviceCount, n, dimX, numberOfHyperplanes, makeDivisions );
			CUDA_CHECK_RETURN( cudaEventRecord( ( cudaEvent_t )m_stop[ device ], 0 ) );

			printf( "Synchronizing...\n" );
			CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
			CUDA_CHECK_RETURN( cudaGetLastError() );
			CUDA_CHECK_RETURN( cudaFreeHost( hostAllocatedMem ) );

			j ? FixLaunchTime( LAUNCH_TIME_STAGE2_SECOND_GROUP, device ) : FixLaunchTime( LAUNCH_TIME_STAGE2_FIRST_GROUP, device );
		}
	}

	if( m_devicesGroups[ 1 ].size() > 0 )
	{
		printf( "Running second stage kernel...\n" );
		int32_t device = m_devicesGroups[ 0 ][ 0 ];
		uint32_t deviceCount = 2;
		CUDA_CHECK_RETURN( cudaSetDevice( device ) );		

		FP** hostAllocatedMem;
		cudaHostAlloc( ( void** )&hostAllocatedMem, deviceCount * sizeof( FP* ), cudaHostAllocDefault );
		for( uint32_t i = 0; i < deviceCount; i++ )
			hostAllocatedMem[ i ] = m_hyperplanesDevPtr[ m_devicesGroups[ 0 ][ i ] ];

		uint32_t srcDevice = m_devicesGroups[ 1 ][ 0 ];
		CUDA_CHECK_RETURN( cudaMemcpyPeer( hostAllocatedMem[ 1 ], m_devicesGroups[ 0 ][ 1 ], m_hyperplanesDevPtr[ srcDevice ], srcDevice, m_hyperplanesArraySize * sizeof( FP ) ) );

		getGridAndBlockDim( numberOfHyperplanes, gridDim, blockDim, device );
		CUDA_CHECK_RETURN( cudaEventRecord( ( cudaEvent_t )m_start[ device ], 0 ) );
		SecondStageKernel<<< gridDim, blockDim >>>( hostAllocatedMem, deviceCount, n, dimX, numberOfHyperplanes, true );
		CUDA_CHECK_RETURN( cudaEventRecord( ( cudaEvent_t )m_stop[ device ], 0 ) );

		CUDA_CHECK_RETURN( cudaMemcpyPeer( m_hyperplanesDevPtr[ srcDevice ], srcDevice, hostAllocatedMem[ 0 ], m_devicesGroups[ 0 ][ 0 ], m_hyperplanesArraySize * sizeof( FP ) ) );

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

	printf( "Running third stage kernel...\n" );
	// циклы объединять не стоит, иначе kernel-ы будут запускаться последовательно
	for( uint32_t j = 0; j < 2; j++ )
		for( uint32_t i = 0; i < m_devicesGroups[ j ].size(); i++ )
		{
			uint32_t device = m_devicesGroups[ j ][ i ];
			CUDA_CHECK_RETURN( cudaSetDevice( device ) );

			// copy hyperplanes from first device to others
			if( i != 0 )
			{
				uint32_t lastDevice = m_devicesGroups[ j ][ i - 1 ];
				CUDA_CHECK_RETURN( cudaEventRecord( ( cudaEvent_t )m_start[ device ], 0 ) );
				CUDA_CHECK_RETURN( cudaMemcpyPeer( m_hyperplanesDevPtr[ device ], device, m_hyperplanesDevPtr[ lastDevice ], lastDevice, m_hyperplanesArraySize * sizeof( FP ) ) );
				CUDA_CHECK_RETURN( cudaEventRecord( ( cudaEvent_t )m_stop[ device ], 0 ) );
			}

		}

	Synchronize( LAUNCH_TIME_COPY_HYPERPLANES );

	for( uint32_t j = 0; j < 2; j++ )
		for( uint32_t i = 0; i < m_devicesGroups[ j ].size(); i++ )
		{
			uint32_t device = m_devicesGroups[ j ][ i ];
			CUDA_CHECK_RETURN( cudaSetDevice( device ) );

			uint32_t pointsPerCurrentDevice = CalcPointsNumberPerDevice( device, deviceCount );

			getGridAndBlockDim( pointsPerCurrentDevice, gridDim, blockDim, device );
			CUDA_CHECK_RETURN( cudaEventRecord( ( cudaEvent_t )m_start[ device ], 0 ) );
			ThirdStageKernel<<< gridDim, blockDim >>>( m_hyperplanesDevPtr[ device ], m_pointsDevPtr[ device ], n, dimX, numberOfHyperplanes, pointsPerCurrentDevice );
			CUDA_CHECK_RETURN( cudaEventRecord( ( cudaEvent_t )m_stop[ device ], 0 ) );
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
		for( uint32_t i = 0; i < m_devicesGroups[ j ].size(); i++ )
		{
			uint32_t device = m_devicesGroups[ j ][ i ];
			CUDA_CHECK_RETURN( cudaSetDevice( device ) );

			//
			CUDA_CHECK_RETURN( cudaEventRecord( ( cudaEvent_t )m_start[ device ], 0 ) );

			//
			uint64_t arrayOffset = m_pointsChunksPerDevice * BLOCK_DIM * device * n;

			uint64_t bytesCount = CalcPointsNumberPerDevice( device, deviceCount ) * n * sizeof( FP );
			printf( "Copying result from GPU%u, %llu bytes\n", device, bytesCount );
			CUDA_CHECK_RETURN( cudaMemcpy( m_points + arrayOffset, m_pointsDevPtr[ device ], bytesCount, cudaMemcpyDeviceToHost ) );			

			//
			CUDA_CHECK_RETURN( cudaEventRecord( ( cudaEvent_t )m_stop[ device ], 0 ) );
		}

	Synchronize( LAUNCH_TIME_DTOH );

	// ???
	uint32_t funcSize = ( uint32_t )size();
	clear();
	printf( "Storing result...\n" );	
	FPVector x( dimX );
	for( uint32_t i = 0; i < funcSize; i++ )
	{
		uint64_t offsetToPoint = ( i - i % BLOCK_DIM ) * n + ( i % BLOCK_DIM );

		for( uint32_t j = 0; j < dimX; j++ )
			x[ j ] = m_points[ offsetToPoint + j * BLOCK_DIM ];

		define( x ) = m_points[ offsetToPoint + ( n - 1 ) * BLOCK_DIM ];
	}
}


//
void ScalarFunction::makeConvexGPU( const uint32_t& dimX, const uint32_t& numberOfPoints )
{
	BOOST_STATIC_ASSERT( sizeof( cudaEvent_t ) == sizeof( m_start[ 0 ] ) );
	BOOST_STATIC_ASSERT( sizeof( cudaEvent_t ) == sizeof( m_stop[ 0 ] ) );

	//
	for( uint32_t i = 0; i < LAUNCH_TIME_COUNT; i++ )
		for( uint32_t j = 0; j < MAX_GPU_COUNT; j++ )
			m_launchTime[ i ][ j ] = 0.0f;

	//
	if( dimX == 0 || numberOfPoints == 0 )
		return;
	
	FP dFi = PI / ( numberOfPoints - 1 );

	uint32_t n = dimX + 1; // space dimension

	uint32_t numberOfHyperplanes = ( uint32_t )powf( ( float )numberOfPoints, ( float )dimX );

	// first x0.. x(n - 2) elements are independent vars. in 2D it will be x
	// x(n - 1) element dependent var. . in 2D it will be y
	// xn - constant, represents distance between O and hyperplane
	m_hyperplanesArraySize = ( numberOfHyperplanes + ( ( numberOfHyperplanes % BLOCK_DIM == 0 ) ? 0 : BLOCK_DIM - numberOfHyperplanes % BLOCK_DIM ) ) * ( n + 1 );
	m_hyperplanes = new FP[ m_hyperplanesArraySize ];

	uint32_t pointsNum = ( ( uint32_t )size() + ( ( ( uint32_t )size() % BLOCK_DIM == 0 ) ? 0 : BLOCK_DIM - ( uint32_t )size() % BLOCK_DIM ) );
	m_pointsArraySize = pointsNum * n;
	m_points = new FP[ m_pointsArraySize ];

	printf( "Number of hyperplanes: %u\n", numberOfHyperplanes );
	printf( "Number of points: %u, unused: %u\n", pointsNum, pointsNum - size() );

	printf( "Memory allocated for hyperplanes: %llu\n", m_hyperplanesArraySize * sizeof( FP ) );
	printf( "Memory allocated for points: %llu\n", m_pointsArraySize * sizeof( FP ) );

	CopyData( dimX );

	InitHyperplanes( dimX, numberOfHyperplanes, dFi );

	uint32_t neededDeviceNumber = pointsNum / MAX_THREADS_PER_DEVICE;
	if( neededDeviceNumber == 0 ) neededDeviceNumber = 1;
	neededDeviceNumber = neededDeviceNumber > MAX_GPU_COUNT ? MAX_GPU_COUNT : neededDeviceNumber;

	uint32_t deviceCount = PrepareDevices( neededDeviceNumber );

	// общее кол-во чанков из точек
	m_pointsChunksNumber = pointsNum / BLOCK_DIM;
	// чанков на одну видеокарту
	m_pointsChunksPerDevice = m_pointsChunksNumber / deviceCount;
	// чанков на последнюю видеокарту
	m_pointsChunksForLastDevice = m_pointsChunksNumber - m_pointsChunksPerDevice * ( deviceCount - 1 );

	//
	DeviceMemoryPreparing( n, deviceCount );

	//
	FirstStage( dimX, numberOfHyperplanes, deviceCount );
	SecondStage( dimX, numberOfHyperplanes );
	ThirdStage( dimX, numberOfHyperplanes, deviceCount );
	GetResult( dimX, deviceCount );

	//
	for( uint32_t j = 0; j < 2; j++ )
		for( uint32_t i = 0; i < m_devicesGroups[ j ].size(); i++ )
		{
			uint32_t device = m_devicesGroups[ j ][ i ];
			CUDA_CHECK_RETURN( cudaSetDevice( device ) ); 

			CUDA_CHECK_RETURN( cudaFree( ( void* )m_hyperplanesDevPtr[ device ] ) );
			CUDA_CHECK_RETURN( cudaFree( ( void* )m_pointsDevPtr[ device ] ) );
			//
			CUDA_CHECK_RETURN( cudaDeviceReset() );
			CUDA_CHECK_RETURN( cudaGetLastError() );
		}

	delete[] m_hyperplanes;
	delete[] m_points;

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
			printf("%12f | ", m_launchTime[ i ][ j ]);
		}
		printf("\n");
	}
}
