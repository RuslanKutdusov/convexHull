#include <iostream>
#include "convexHull.hpp"
#include "Image.hpp"

#define PI 3.1415926f

void makeConvexHull( FunctionOfAny< std::vector< FP >, FP >& function, const int32_t& stepNumber )
{
	if( function.size() == 0 )
		return;
	
	FP dFi = PI / stepNumber;

	int32_t n = function.begin()->first.size() + 1; // space dimension

	std::vector< FP > normal( n );

	// first x0.. x(n - 2) elements are independent vars
	// x(n - 1) element dependent var
	// xn - constant, represents distance between O and hyperplane
	std::vector< std::vector< FP > >hyperplanes( stepNumber );

	FP fi = 0.0;

	for( int32_t i = 0; i < stepNumber; i++ )
	{
		hyperplanes[ i ] = std::vector< FP >( n + 1 );

		fi += dFi;

		normal[ 0 ] = cos( fi );
		normal[ 1 ] = sin( fi );

		hyperplanes[ i ][ n ] = 0.0f; 
		
		for( FunctionOfAny< std::vector< FP >, FP >::const_iterator iter = function.begin(); iter != function.end(); ++iter )
		{
			FP d = 0.0;

			// dot product of point and normal - distance
			for( int32_t j = 0; j < n - 1; j++ )
				d += iter->first[ j ] * normal[ j ];
			d += iter->second * normal[ n - 1 ];

			if( d > hyperplanes[ i ][ n ] )
			{
				for( int32_t j = 0; j < n; j++ )
					hyperplanes[ i ][ j ] = normal[ j ];

				hyperplanes[ i ][ n ] = d;
			}
		}
	}

	 for( FunctionOfAny< std::vector< FP >, FP >::iterator iter = function.begin(); iter != function.end(); ++iter )
	 {
	 	for( int32_t i = 0; i < stepNumber; i++ )
	 	{
	 		FP val = 0.0;
	 		// xi - iter->first
	 		// Ni - hyperplane normal
	 		// val = x(n - 1) = ( -N0*x0 - N1*x1 - ... - N(n - 2)*x(n - 2) + xn ) / N(n - 1)
	 		for( int32_t j = 0; j < n - 1; j++ )
	 			val -= iter->first[ j ] * hyperplanes[ i ][ j ];
	 		val += hyperplanes[ i ][ n ];
	 		val /= hyperplanes[ i ][ n - 1 ];

	 		if( i == 0 )
	 		{
	 			iter->second = val;
	 			continue;
	 		}

	 		if( val < iter->second )
	 			iter->second = val;
	 	}
	 }
}
