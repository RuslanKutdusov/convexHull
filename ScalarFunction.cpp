#include "ScalarFunction.hpp"

#include <boost/thread.hpp>
#include <boost/shared_ptr.hpp>

void ScalarFunction::makeConvex( const uint32_t& dimX, const uint32_t& numberOfPoints )
{
	if( dimX == 0 )
		return;
	
	FP dFi = PI / ( numberOfPoints - 1 );

	uint32_t n = dimX + 1; // space dimension

	uint32_t numberOfHyperplanes = pow( numberOfPoints, n - 1 );

	// first x0.. x(n - 2) elements are independent vars. in 2D it will be x, contains normal
	// x(n - 1) element dependent var. . in 2D it will be y
	// xn - constant, represents distance between O and hyperplane
	std::vector< FPVector > hyperplanes( numberOfHyperplanes, FPVector( n + 1 ) );

	FPVector fi( dimX, 0.0 );

	for( uint32_t i = 0; i < numberOfHyperplanes; i++ )
	{
		for( uint32_t j = 0; j < n; j++ )
		{
			hyperplanes[ i ][ j ] = 1.0;
			for( uint32_t k = 0; k < j; k++ )
				hyperplanes[ i ][ j ] *= sin( fi[ k ] );

			if( j != n - 1 )
				hyperplanes[ i ][ j ] *= cos( fi[ j ] );
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

		hyperplanes[ i ][ n ] = 0.0; 
		
		for( const_iterator iter = begin(); iter != end(); ++iter )
		{
			FP d = 0.0;

			// dot product of point and normal is distance
			for( uint32_t j = 0; j < dimX; j++ )
				d += iter->first[ j ] * hyperplanes[ i ][ j ];
			d += iter->second * hyperplanes[ i ][ n - 1 ];

			if( d > hyperplanes[ i ][ n ] )
				hyperplanes[ i ][ n ] = d;
		}
	}

	for( iterator iter = begin(); iter != end(); ++iter )
	{
		FP funcVal = iter->second;
		for( uint32_t i = 0; i < numberOfHyperplanes; i++ )
		{
			FP val = 0.0;
			// xi - iter->first
			// Ni - hyperplane normal
			// val = x(n - 1) = ( -N0*x0 - N1*x1 - ... - N(n - 2)*x(n - 2) + xn ) / N(n - 1)
			for( uint32_t j = 0; j < dimX; j++ )
				val -= iter->first[ j ] * hyperplanes[ i ][ j ];
			val += hyperplanes[ i ][ n ];
			val /= hyperplanes[ i ][ n - 1 ] + EPSILON;

			if( i == 0 )
			{
				iter->second = val;
				continue;
			}

			if( val < iter->second && val >= funcVal )
				iter->second = val;
		}
	}
}


//
void thread1( std::vector< FPVector >& hyperplanes, std::vector< FPVector >& fi, 
			const uint32_t& startIndex, const uint32_t& taskSize, const uint32_t& n, 
			ScalarFunction* func )
{
	uint32_t dimX = n - 1;

	for( uint32_t i = startIndex; i < startIndex + taskSize; i++ )
	{
		for( uint32_t j = 0; j < n; j++ )
		{
			hyperplanes[ i ][ j ] = 1.0;
			for( uint32_t k = 0; k < j; k++ )
				hyperplanes[ i ][ j ] *= sin( fi[ i ][ k ] );

			if( j != n - 1 )
				hyperplanes[ i ][ j ] *= cos( fi[ i ][ j ] );
		}

		hyperplanes[ i ][ n ] = 0.0; 
		
		for( ScalarFunction::const_iterator iter = func->begin(); iter != func->end(); ++iter )
		{
			FP d = 0.0;

			// dot product of point and normal is distance
			for( uint32_t j = 0; j < dimX; j++ )
				d += iter->first[ j ] * hyperplanes[ i ][ j ];
			d += iter->second * hyperplanes[ i ][ n - 1 ];

			if( d > hyperplanes[ i ][ n ] )
				hyperplanes[ i ][ n ] = d;
		}
	}
}


void thread2( std::vector< FPVector >& hyperplanes, std::vector< const FPVector* >& points, 
			const uint32_t& startIndex, const uint32_t& taskSize, const uint32_t& n, 
			ScalarFunction* func )
{
	uint32_t dimX = n - 1;

	for( uint32_t k = startIndex; k < startIndex + taskSize; k++ )
	{
		FP funcVal = func->at( *points[ k ] );
		FP ret = funcVal;

		for( uint32_t i = 0; i < hyperplanes.size(); i++ )
		{
			FP val = 0.0;
			// xi - iter->first
			// Ni - hyperplane normal
			// val = x(n - 1) = ( -N0*x0 - N1*x1 - ... - N(n - 2)*x(n - 2) + xn ) / N(n - 1)
			for( uint32_t j = 0; j < dimX; j++ )
				val -= ( *points[ k ] )[ j ] * hyperplanes[ i ][ j ];
			val += hyperplanes[ i ][ n ];
			val /= hyperplanes[ i ][ n - 1 ] + EPSILON;

			if( i == 0 )
			{
				ret = val;
				continue;
			}

			if( val < ret && val >= funcVal )
				ret = val;
		}

		func->define( *points[ k ] ) = ret;
	}
}


//
void JoinThread( boost::thread* thread )
{
	thread->join();
	delete thread;
}


//
void ScalarFunction::makeConvexMultiThread( const uint32_t& dimX, const uint32_t& numberOfPoints, const uint32_t& jobs )
{
	if( dimX == 0 )
		return;
	
	FP dFi = PI / ( numberOfPoints - 1 );

	uint32_t n = dimX + 1; // space dimension

	uint32_t numberOfHyperplanes = pow( numberOfPoints, n - 1 );

	FPVector normal( n );

	// first x0.. x(n - 2) elements are independent vars. in 2D it will be x
	// x(n - 1) element dependent var. . in 2D it will be y
	// xn - constant, represents distance between O and hyperplane
	std::vector< FPVector > hyperplanes( numberOfHyperplanes, FPVector( n + 1 ) );

	// prepare Fi for all hyperplanes
	std::vector< FPVector > fi( numberOfHyperplanes, FPVector( dimX, 0.0 ) );
	for( uint32_t i = 1; i < numberOfHyperplanes; i++ )
	{
		bool shift = true;
		fi[ i ] = fi[ i - 1 ];
		for( uint32_t k = 0; ( k < dimX ) && shift; k++ )
		{
			if( fabs( fi[ i - 1 ][ k ] - PI ) <= EPSILON )
			{
				fi[ i ][ k ] = 0.0;
				shift = true;	
			}
			else
			{
				fi[ i ][ k ] = fi[ i - 1 ][ k ] + dFi;
				shift = false;
			}

			if( fi[ i ][ k ] - PI > EPSILON )
				fi[ i ][ k ] = PI;
		}
	}

	{
		std::vector< boost::shared_ptr< boost::thread > > threads( jobs );

		uint32_t tasksNumber = numberOfHyperplanes;

		uint32_t tasksPerThread = tasksNumber / jobs;

		for( uint32_t i = 0; i < jobs; i++ )
		{
			uint32_t tasksNumberForThread = i == jobs - 1 ? tasksNumber - i * tasksPerThread : tasksPerThread;
			threads[ i ] = boost::shared_ptr< boost::thread >( new boost::thread( thread1, boost::ref( hyperplanes ), boost::ref( fi ), i * tasksPerThread, 
																tasksNumberForThread, n, this ),
																&JoinThread );
		}
	}

	std::vector< const FPVector* > points( size() );

	uint32_t i = 0;
	for( iterator iter = begin(); iter != end(); ++iter, i++ )
		points[ i ] = &iter->first;

	{
		std::vector< boost::shared_ptr< boost::thread > > threads( jobs );

		uint32_t tasksNumber = size();

		uint32_t tasksPerThread = tasksNumber / jobs;

		for( uint32_t i = 0; i < jobs; i++ )
		{
			uint32_t tasksNumberForThread = i == jobs - 1 ? tasksNumber - i * tasksPerThread : tasksPerThread;
			threads[ i ] = boost::shared_ptr< boost::thread >( new boost::thread( thread2, boost::ref( hyperplanes ), boost::ref( points ), i * tasksPerThread, 
																tasksNumberForThread, n, this ),
																&JoinThread );
		}
	}
}

//
#ifdef GPU
#include "gpu.hpp"

void ScalarFunction::makeConvexGPU( const uint32_t& dimX, const uint32_t& numberOfPoints )
{
	gpu::makeConvex( *this, dimX, numberOfPoints );
}

#endif
