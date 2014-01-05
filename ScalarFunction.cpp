#include "ScalarFunction.hpp"

void ScalarFunction::makeConvex( const size_t& dimX, const size_t& numberOfPoints )
{
	if( dimX == 0 )
		return;
	
	FP dFi = M_PI / ( numberOfPoints - 1 );

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
		for( size_t k = 0; ( k < fi.size() ) && shift; k++ )
		{
			fi[ k ] += dFi;
			if( fi[ k ] - M_PI > EPSILON )
			{
				fi[ k ] = 0.0;
				shift = true;
			}
			else
			{
				shift = false;
			}
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
