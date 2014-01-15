#include "ScalarFunction.hpp"

#include <boost/thread.hpp>
#include <boost/shared_ptr.hpp>

void ScalarFunction::makeConvex( const size_t& dimX, const size_t& numberOfPoints )
{
	if( dimX == 0 )
		return;
	
	FP dFi = PI / ( numberOfPoints - 1 );

	size_t n = dimX + 1; // space dimension

	size_t numberOfHyperplanes = pow( numberOfPoints, n - 1 );

	FPVector normal( n );

	// first x0.. x(n - 2) elements are independent vars. in 2D it will be x
	// x(n - 1) element dependent var. . in 2D it will be y
	// xn - constant, represents distance between O and hyperplane
	std::vector< FPVector > hyperplanes( numberOfHyperplanes, FPVector( n + 1 ) );

	FPVector fi( dimX, 0.0 );

	for( size_t i = 0; i < numberOfHyperplanes; i++ )
	{
		for( size_t j = 0; j < n; j++ )
		{
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

		hyperplanes[ i ][ n ] = 0.0; 
		
		for( const_iterator iter = begin(); iter != end(); ++iter )
		{
			FP d = 0.0;

			// dot product of point and normal is distance
			for( size_t j = 0; j < dimX; j++ )
				d += iter->first[ j ] * normal[ j ];
			d += iter->second * normal[ n - 1 ];

			if( d > hyperplanes[ i ][ n ] )
			{
				for( size_t j = 0; j < n; j++ )
					hyperplanes[ i ][ j ] = normal[ j ];

				hyperplanes[ i ][ n ] = d;
			}
		}
	}

	for( iterator iter = begin(); iter != end(); ++iter )
	{
		FP funcVal = iter->second;
		for( size_t i = 0; i < numberOfHyperplanes; i++ )
		{
			FP val = 0.0;
			// xi - iter->first
			// Ni - hyperplane normal
			// val = x(n - 1) = ( -N0*x0 - N1*x1 - ... - N(n - 2)*x(n - 2) + xn ) / N(n - 1)
			for( size_t j = 0; j < dimX; j++ )
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
			const size_t& startIndex, const size_t& taskSize, const size_t& n, 
			ScalarFunction* func )
{
	size_t dimX = n - 1;
	FPVector normal( n );

	for( size_t i = startIndex; i < startIndex + taskSize; i++ )
	{
		for( size_t j = 0; j < n; j++ )
		{
			normal[ j ] = 1.0;
			for( size_t k = 0; k < j; k++ )
				normal[ j ] *= sin( fi[ i ][ k ] );

			if( j != n - 1 )
				normal[ j ] *= cos( fi[ i ][ j ] );
		}

		hyperplanes[ i ][ n ] = 0.0; 
		
		for( ScalarFunction::const_iterator iter = func->begin(); iter != func->end(); ++iter )
		{
			FP d = 0.0;

			// dot product of point and normal is distance
			for( size_t j = 0; j < dimX; j++ )
				d += iter->first[ j ] * normal[ j ];
			d += iter->second * normal[ n - 1 ];

			if( d > hyperplanes[ i ][ n ] )
			{
				for( size_t j = 0; j < n; j++ )
					hyperplanes[ i ][ j ] = normal[ j ];

				hyperplanes[ i ][ n ] = d;
			}
		}
	}
}


void thread2( std::vector< FPVector >& hyperplanes, std::vector< const FPVector* >& points, 
			const size_t& startIndex, const size_t& taskSize, const size_t& n, 
			ScalarFunction* func )
{
	size_t dimX = n - 1;

	for( size_t k = startIndex; k < startIndex + taskSize; k++ )
	{
		FP funcVal = func->at( *points[ k ] );
		FP ret = funcVal;

		for( size_t i = 0; i < hyperplanes.size(); i++ )
		{
			FP val = 0.0;
			// xi - iter->first
			// Ni - hyperplane normal
			// val = x(n - 1) = ( -N0*x0 - N1*x1 - ... - N(n - 2)*x(n - 2) + xn ) / N(n - 1)
			for( size_t j = 0; j < dimX; j++ )
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
void ScalarFunction::makeConvexMultiThread( const size_t& dimX, const size_t& numberOfPoints, const size_t& jobs )
{
	if( dimX == 0 )
		return;
	
	FP dFi = PI / ( numberOfPoints - 1 );

	size_t n = dimX + 1; // space dimension

	size_t numberOfHyperplanes = pow( numberOfPoints, n - 1 );

	FPVector normal( n );

	// first x0.. x(n - 2) elements are independent vars. in 2D it will be x
	// x(n - 1) element dependent var. . in 2D it will be y
	// xn - constant, represents distance between O and hyperplane
	std::vector< FPVector > hyperplanes( numberOfHyperplanes, FPVector( n + 1 ) );

	// prepare Fi for all hyperplanes
	std::vector< FPVector > fi( numberOfHyperplanes, FPVector( dimX, 0.0 ) );
	for( size_t i = 1; i < numberOfHyperplanes; i++ )
	{
		bool shift = true;
		fi[ i ] = fi[ i - 1 ];
		for( size_t k = 0; ( k < dimX ) && shift; k++ )
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

		size_t tasksNumber = numberOfHyperplanes;

		size_t tasksPerThread = tasksNumber / jobs;

		for( size_t i = 0; i < jobs; i++ )
		{
			size_t tasksNumberForThread = i == jobs - 1 ? tasksNumber - i * tasksPerThread : tasksPerThread;
			threads[ i ] = boost::shared_ptr< boost::thread >( new boost::thread( thread1, boost::ref( hyperplanes ), boost::ref( fi ), i * tasksPerThread, 
																tasksNumberForThread, n, this ),
																&JoinThread );
		}
	}

	std::vector< const FPVector* > points( size() );

	size_t i = 0;
	for( iterator iter = begin(); iter != end(); ++iter, i++ )
		points[ i ] = &iter->first;

	{
		std::vector< boost::shared_ptr< boost::thread > > threads( jobs );

		size_t tasksNumber = size();

		size_t tasksPerThread = tasksNumber / jobs;

		for( size_t i = 0; i < jobs; i++ )
		{
			size_t tasksNumberForThread = i == jobs - 1 ? tasksNumber - i * tasksPerThread : tasksPerThread;
			threads[ i ] = boost::shared_ptr< boost::thread >( new boost::thread( thread2, boost::ref( hyperplanes ), boost::ref( points ), i * tasksPerThread, 
																tasksNumberForThread, n, this ),
																&JoinThread );
		}
	}
}

//
#ifdef GPU
#include "gpu.hpp"

void ScalarFunction::makeConvexGPU( const size_t& dimX, const size_t& numberOfPoints )
{
	gpu::makeConvex( *this, dimX, numberOfPoints );
}

#endif
