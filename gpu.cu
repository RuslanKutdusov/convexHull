#include "gpu.hpp"

#include <stdio.h>
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

// TODO: cuda arrays??
texture< FP, 1, cudaReadModeElementType > g_texturePoints;
texture< FP, 1, cudaReadModeElementType > g_textureVals;
texture< FP, 1, cudaReadModeElementType > g_textureHyperplanes;


//
__global__ void kernel1( FP* hyperplanes, FP* points, FP* vals, const size_t n, size_t numberOfHyperplanes, size_t numberOfPoints )
{
	size_t hyperplaneIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if( hyperplaneIndex >= numberOfHyperplanes )
		return;

	size_t dimX = n - 1;
	// now its offset to hyperplane in 'hyperplanes' array, to remove redundant multiplication at every string
	size_t offsetToHyperplane = hyperplaneIndex * ( n + 1 );

	FP resultDistance = 0.0;
	
	for( size_t k = 0; k < numberOfPoints; k++ )
	{
		FP d = 0.0;

		// dot product of point and normal is distance
		for( size_t j = 0; j < dimX; j++ ) 
			d += points[ k * dimX + j ] * hyperplanes[ offsetToHyperplane + j ]; // TODO: shared memory in k loop?
		d += vals[ k ] * hyperplanes[ offsetToHyperplane + n - 1 ]; 

		if( d > resultDistance )
			resultDistance = d;
	}

	hyperplanes[ offsetToHyperplane + n ] = resultDistance;
}


// TODO: pair?
__global__ void kernel1_1( FP** hyperplanes, int deviceCount, size_t n, size_t numberOfHyperplanes )
{
	size_t hyperplaneIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if( hyperplaneIndex >= numberOfHyperplanes )
		return;

	size_t offset = hyperplaneIndex * ( n + 1 ) + n;

	FP resultDistance = hyperplanes[ 0 ][ offset ];
	for( int i = 1; i < deviceCount; i++ )
	{
		if( hyperplanes[ i ][ offset ] > resultDistance )
			resultDistance = hyperplanes[ i ][ offset ];
	}
	hyperplanes[ 0 ][ offset ] = resultDistance;
}


// sending dimX as argument is to reduce registers usage
__global__ void kernel2( FP* hyperplanes, FP* points, FP* vals, size_t n, size_t dimX, size_t numberOfHyperplanes, size_t numberOfPoints )
{
	size_t k = blockIdx.x * blockDim.x + gridDim.x * blockDim.x * blockIdx.y + threadIdx.x;
	if( k >= numberOfPoints )
		return;

	FP funcVal = vals[ k ];
	FP convexVal = funcVal;

	for( size_t i = 0; i < numberOfHyperplanes; i++ )
	{
		FP val = 0.0;
		size_t offsetToHyperplane = i * ( n + 1 );
		// xi - iter->first
		// Ni - hyperplane normal
		// val = x(n - 1) = ( -N0*x0 - N1*x1 - ... - N(n - 2)*x(n - 2) + xn ) / N(n - 1)
		for( size_t j = 0; j < dimX; j++ )
			val -= points[ k * dimX + j ] * hyperplanes[ offsetToHyperplane + j ];
		val += hyperplanes[ offsetToHyperplane + n ];
		val /= hyperplanes[ offsetToHyperplane + n - 1 ] + EPSILON;

		if( i == 0 )
		{
			convexVal = val;
			continue;
		}

		if( val < convexVal && val >= funcVal )
			convexVal = val;
	}

	vals[ k ] = convexVal;
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
__host__ void makeConvex( ScalarFunction& func, const size_t& dimX, const size_t& numberOfPoints )
{
	if( dimX == 0 )
		return;
	
	FP dFi = PI / ( numberOfPoints - 1 );

	size_t n = dimX + 1; // space dimension

	size_t numberOfHyperplanes = pow( numberOfPoints, n - 1 );

	// first x0.. x(n - 2) elements are independent vars. in 2D it will be x
	// x(n - 1) element dependent var. . in 2D it will be y
	// xn - constant, represents distance between O and hyperplane
	size_t hyperplanesArraySize = numberOfHyperplanes * ( n + 1 );
	size_t hyperplanesArrayLength = hyperplanesArraySize * sizeof( FP );
	FP* hyperplanes = new FP[ hyperplanesArraySize ];

	size_t pointsArraySize = dimX * func.size();
	FP* points = new FP[ pointsArraySize ];

	size_t valsArraySize = func.size();
	FP* vals = new FP[ valsArraySize ];

	{
		size_t i = 0;
		for( ScalarFunction::iterator iter = func.begin(); iter != func.end(); ++iter, i++ )
		{
			for( size_t j = 0; j < dimX; j++ )
				points[ i * dimX + j ] = iter->first[ j ];

			vals[ i ] = iter->second;
		}
	}

	FPVector fi( dimX, 0.0 );

	for( size_t i = 0; i < numberOfHyperplanes; i++ )
	{
		for( size_t j = 0; j < n; j++ )
		{
			FP* normal = &hyperplanes[ i * ( n + 1 ) ];

			normal[ j ] = 1.0;
			for( size_t k = 0; k < j; k++ )
				normal[ j ] *= sin( fi[ k ] );

			if( j != n - 1 )
				normal[ j ] *= cos( fi[ j ] );
		}

		// not good enough
		bool shift = true;
		for( size_t k = 0; ( k < dimX ) && shift; k++ )
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
		CUDA_CHECK_RETURN( cudaDeviceEnablePeerAccess( usedDevices[ i ], 0 ) );

	printf( "Used device count: %d\n", deviceCount );

	const size_t pointsPerDevice = func.size() / deviceCount;
	FP* d_hyperplanes[ MAX_GPU_COUNT ];
	FP* d_points[ MAX_GPU_COUNT ];
	FP* d_vals[ MAX_GPU_COUNT ];
	dim3 gridDim, blockDim;

	//
	printf( "Running first kernel...\n" );
	for( int i = 0; i < deviceCount; i++ )
	{
		int device = usedDevices[ i ];
		CUDA_CHECK_RETURN( cudaSetDevice( device ) );

		//
		CUDA_CHECK_RETURN( cudaMalloc( &d_hyperplanes[ i ], hyperplanesArrayLength ) );
		//CUDA_CHECK_RETURN( cudaBindTexture( NULL, g_textureHyperplanes[ i ], d_hyperplanes[ i ], hyperplanesArrayLength ) );

		if( i == 0 )
		{
			CUDA_CHECK_RETURN( cudaMemcpy( d_hyperplanes[ i ], hyperplanes, hyperplanesArrayLength, cudaMemcpyHostToDevice ) );
		}
		else
		{
			// TODO: smart copying, pair
			int lastDevice = usedDevices[ i - 1 ];
			CUDA_CHECK_RETURN( cudaMemcpyPeer( d_hyperplanes[ i ], device, d_hyperplanes[ i - 1 ], lastDevice, hyperplanesArrayLength ) );
		}

		int arrayOffset = pointsPerDevice * i;
		int pointsPerCurrentDevice = pointsPerDevice;
		if( i == deviceCount - 1 )
			pointsPerCurrentDevice = func.size() - pointsPerDevice * i;

		//
		int bytesCount = pointsPerCurrentDevice * dimX * sizeof( FP );
		CUDA_CHECK_RETURN( cudaMalloc( &d_points[ i ], bytesCount ) );
		CUDA_CHECK_RETURN( cudaMemcpy( d_points[ i ], points + arrayOffset * dimX, bytesCount, cudaMemcpyHostToDevice ) );
		//CUDA_CHECK_RETURN( cudaBindTexture( NULL, g_texturePoints[ i ], d_points[ i ], bytesCount ) );

		//
		bytesCount = pointsPerCurrentDevice * sizeof( FP );
		CUDA_CHECK_RETURN( cudaMalloc( &d_vals[ i ], bytesCount ) );
		CUDA_CHECK_RETURN( cudaMemcpy( d_vals[ i ], vals + arrayOffset, bytesCount, cudaMemcpyHostToDevice ) );
		//CUDA_CHECK_RETURN( cudaBindTexture( NULL, g_textureVals[ i ], d_vals, bytesCount ) );

		// run first kernel
		getGridAndBlockDim( numberOfHyperplanes, gridDim, blockDim, device );
		kernel1<<< gridDim, blockDim >>>( d_hyperplanes[ i ], d_points[ i ], d_vals[ i ], n, numberOfHyperplanes, pointsPerCurrentDevice );
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

	//
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
			CUDA_CHECK_RETURN( cudaMemcpyPeer( d_hyperplanes[ i ], device, d_hyperplanes[ i - 1 ], lastDevice, hyperplanesArrayLength ) );
		}

		int pointsPerCurrentDevice = pointsPerDevice;
		if( i == deviceCount - 1 )
			pointsPerCurrentDevice = func.size() - pointsPerDevice * i;

		getGridAndBlockDim( pointsPerCurrentDevice, gridDim, blockDim, device );
		kernel2<<< gridDim, blockDim >>>( d_hyperplanes[ i ], d_points[ i ], d_vals[ i ], n, dimX, numberOfHyperplanes, pointsPerCurrentDevice );
	}

	//
	printf( "Synchronizing...\n" );
	for( int i = 0; i < deviceCount; i++ )
	{
		int device = usedDevices[ i ];
		CUDA_CHECK_RETURN( cudaSetDevice( device ) );
		CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
		CUDA_CHECK_RETURN( cudaGetLastError() );

		int arrayOffset = pointsPerDevice * i;
		int pointsPerCurrentDevice = pointsPerDevice;
		if( i == deviceCount - 1 )
			pointsPerCurrentDevice = func.size() - pointsPerDevice * i;

		//
		int bytesCount = pointsPerCurrentDevice * sizeof( FP );
		printf( "Copying result from GPU%d, %d bytes\n", device, bytesCount );
		CUDA_CHECK_RETURN( cudaMemcpy( vals + arrayOffset, d_vals[ i ], bytesCount, cudaMemcpyDeviceToHost ) );
	}

	// int device = usedDevices[ 0 ];
	// CUDA_CHECK_RETURN( cudaSetDevice( device ) );

	// FP* d_points_;
	// CUDA_CHECK_RETURN( cudaMalloc( &d_points_, pointsArrayLength ) );
	// CUDA_CHECK_RETURN( cudaMemcpy( d_points_, points, pointsArrayLength, cudaMemcpyHostToDevice ) );

	// FP* d_vals_;
	// CUDA_CHECK_RETURN( cudaMalloc( &d_vals_, valsArrayLength ) );
	// CUDA_CHECK_RETURN( cudaMemcpy( d_vals_, vals, valsArrayLength, cudaMemcpyHostToDevice ) );

	// // run second kernel
	// getGridAndBlockDim( func.size(), gridDim, blockDim, device );
	// kernel2<<< gridDim, blockDim >>>( d_hyperplanes[ 0 ], d_points_, d_vals_, n, dimX, numberOfHyperplanes, func.size() );

	// CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
	// CUDA_CHECK_RETURN( cudaGetLastError() );

	// CUDA_CHECK_RETURN( cudaMemcpy( vals, d_vals_, valsArraySize * sizeof( FP ), cudaMemcpyDeviceToHost ) );
	// CUDA_CHECK_RETURN( cudaGetLastError() );


	// ???
	//func.clear();

	printf( "Storing result...\n" );	
	for( size_t k = 0; k < func.size(); k++ )
	{
		FPVector x( &points[ k * dimX ], &points[ ( k + 1 ) * dimX ] );
		func.define( x ) = vals[ k ];
	}

	for( int i = 0; i < deviceCount; i++ )
	{
		int device = usedDevices[ i ];
		CUDA_CHECK_RETURN( cudaSetDevice( device ) ); 
		CUDA_CHECK_RETURN( cudaFree( ( void* )d_hyperplanes[ i ] ) );
		CUDA_CHECK_RETURN( cudaFree( ( void* )d_points[ i ] ) );
		CUDA_CHECK_RETURN( cudaFree( ( void* )d_vals[ i ] ) );

		//
		CUDA_CHECK_RETURN( cudaDeviceReset() );
		CUDA_CHECK_RETURN( cudaGetLastError() );
	}

	delete[] hyperplanes;
	delete[] points;
	delete[] vals;

	printf( "Done\n" );
}

}