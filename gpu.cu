#include "gpu.hpp"

#include <stdio.h>

#define CUDA_CHECK_RETURN( value ) {											\
	cudaError_t _m_cudaStat = value;										\
	if ( _m_cudaStat != cudaSuccess ) {										\
		fprintf( stderr, "Error '%s' at line %d in file %s\n",					\
				cudaGetErrorString( _m_cudaStat ), __LINE__, __FILE__ );		\
		exit( 1 );															\
	} }

namespace gpu
{

texture< FP, 1, cudaReadModeElementType > g_texturePoints;
texture< FP, 1, cudaReadModeElementType > g_textureVals;
texture< FP, 1, cudaReadModeElementType > g_textureHyperplanes;

__global__ void kernel1( FP* hyperplanes, const size_t n, size_t numberOfHyperplanes, size_t numberOfPoints )
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
			d += tex1Dfetch( g_texturePoints, k * dimX + j ) * hyperplanes[ offsetToHyperplane + j ]; 
		d += tex1Dfetch( g_textureVals, k ) * hyperplanes[ offsetToHyperplane + n - 1 ]; 

		if( d > resultDistance )
			resultDistance = d;
	}

	hyperplanes[ offsetToHyperplane + n ] = resultDistance;
}


// sending dimX as argument is to reduce register usage
__global__ void kernel2( FP* vals, size_t n, size_t dimX, size_t numberOfHyperplanes, size_t numberOfPoints )
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
			val -= tex1Dfetch( g_texturePoints, k * dimX + j ) * tex1Dfetch( g_textureHyperplanes, offsetToHyperplane + j );
		val += tex1Dfetch( g_textureHyperplanes, offsetToHyperplane + n );
		val /= tex1Dfetch( g_textureHyperplanes, offsetToHyperplane + n - 1 ) + EPSILON;

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
void getGridAndBlockDim( int n, dim3& gridDim, dim3& blockDim )
{
	// gpu hardware limits
	const int warpSize = 32;
	//const int smNumber = 2;
	//const int blocksPerSM = 8;
    //const int maxGridSize = smNumber * blocksPerSM;
    const int maxGridDimX = 65535;

    int warpCount = ( n / warpSize ) + ( ( ( n % warpSize ) == 0 ) ? 0 : 1 );
    int warpPerBlock = max( 1, min( 4, warpCount ) );

    int threadCount = warpSize * warpPerBlock;
    int blockCount = ( warpCount / warpPerBlock ) + ( ( ( warpCount % warpPerBlock ) == 0 ) ? 0 : 1 );

    blockDim = dim3( threadCount, 1, 1 );

    gridDim = dim3( blockCount, 1, 1 );

    if( blockCount > maxGridDimX )
	{
		gridDim.x = gridDim.y = sqrt( blockCount );
		if( gridDim.x * gridDim.x < blockCount )
			gridDim.x += 1;
	}
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
	size_t hyperplanesSize = numberOfHyperplanes * ( n + 1 );
	FP* hyperplanes = new FP[ hyperplanesSize ];

	size_t pointsSize = dimX * func.size();
	FP* points = new FP[ pointsSize ];

	size_t valsSize = func.size();
	FP* vals = new FP[ valsSize ];

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

	FP* d_hyperplanes;
	CUDA_CHECK_RETURN( cudaMalloc( &d_hyperplanes, hyperplanesSize * sizeof( FP ) ) );
	CUDA_CHECK_RETURN( cudaMemcpy( d_hyperplanes, hyperplanes, hyperplanesSize * sizeof( FP ), cudaMemcpyHostToDevice ) );
	CUDA_CHECK_RETURN( cudaBindTexture( NULL, g_textureHyperplanes, d_hyperplanes, hyperplanesSize * sizeof( FP ) ) );

	FP* d_points;
	CUDA_CHECK_RETURN( cudaMalloc( &d_points, pointsSize * sizeof( FP ) ) );
	CUDA_CHECK_RETURN( cudaMemcpy( d_points, points, pointsSize * sizeof( FP ), cudaMemcpyHostToDevice ) );
	CUDA_CHECK_RETURN( cudaBindTexture( NULL, g_texturePoints, d_points, pointsSize * sizeof( FP ) ) );

	FP* d_vals;
	CUDA_CHECK_RETURN( cudaMalloc( &d_vals, valsSize * sizeof( FP ) ) );
	CUDA_CHECK_RETURN( cudaMemcpy( d_vals, vals, valsSize * sizeof( FP ), cudaMemcpyHostToDevice ) );
	CUDA_CHECK_RETURN( cudaBindTexture( NULL, g_textureVals, d_vals, valsSize * sizeof( FP ) ) );


	dim3 gridDim, blockDim;

	// run first kernel
	getGridAndBlockDim( numberOfHyperplanes, gridDim, blockDim );
	kernel1<<< gridDim, blockDim >>>( d_hyperplanes, n, numberOfHyperplanes, func.size() );

	CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
	CUDA_CHECK_RETURN( cudaGetLastError() );

	// run second kernel
	getGridAndBlockDim( func.size(), gridDim, blockDim );
	kernel2<<< gridDim, blockDim >>>( d_vals, n, dimX, numberOfHyperplanes, func.size() );

	CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
	CUDA_CHECK_RETURN( cudaGetLastError() );

	CUDA_CHECK_RETURN( cudaMemcpy( vals, d_vals, valsSize * sizeof( FP ), cudaMemcpyDeviceToHost ) );
	CUDA_CHECK_RETURN( cudaGetLastError() );


	// ???
	//func.clear();

	for( size_t k = 0; k < func.size(); k++ )
	{
		FPVector x( &points[ k * dimX ], &points[ ( k + 1 ) * dimX ] );
		func.define( x ) = vals[ k ];
	}

	CUDA_CHECK_RETURN( cudaFree( ( void* )d_hyperplanes ) );
	CUDA_CHECK_RETURN( cudaFree( ( void* )d_points ) );
	CUDA_CHECK_RETURN( cudaFree( ( void* )d_vals ) );

	CUDA_CHECK_RETURN( cudaDeviceReset() );
	CUDA_CHECK_RETURN( cudaGetLastError() );

	delete[] hyperplanes;
	delete[] points;
	delete[] vals;
}

}