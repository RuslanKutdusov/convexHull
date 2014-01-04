#include <math.h>
#include <stdio.h>

#include "Image.hpp"
#include "FunctionOfAny.hpp"
#include "convexHull.hpp"

#include "ScalarFunction.hpp"

void testImage()
{
	float segmentXLength = 2.0f * M_PI + M_PI;
	float segmentYLength = 4.0f;

	Image image( 512, 512 );
	image.clearByWhite();

	float stepX = segmentXLength / image.getWidth();
	float stepY = segmentYLength / image.getHeight();

	float x = -M_PI;
	for( size_t xp = 0; xp < image.getWidth(); xp++ )
	{
		float y = segmentYLength / 2.0f - sin( x );

		size_t yp = y / stepY;

		image.writePixel( xp, yp );

		x += stepX;
	}

	save_png( "out.png", image );
}

void testNormal()
{
	float segmentXLength = 4.0f;
	float segmentYLength = 4.0f;

	Image image( 512, 512 );
	image.clearByWhite();

	float stepX = segmentXLength / image.getWidth();
	float stepY = segmentYLength / image.getHeight();

	float dFi = M_PI / 10.0f;

	for( float fi = 0.0f; fi < M_PI; fi += dFi )
	{
		float x = cos( fi )  + segmentXLength / 2.0f;
		float y = segmentYLength / 2.0f - sin( fi );

		image.writePixel( x / stepX, y / stepY );
	}

	save_png( "out.png", image );
}

void test_makeConvexHull()
{
	float segmentXLength = 2.0 * M_PI;
	float segmentYLength = 20.0f;

	Image image( 512, 512 );
	image.clearByWhite();

	float stepX = segmentXLength / image.getWidth();
	float stepY = segmentYLength / image.getHeight();

	ScalarFunction function;

	FPVector x( 1 );
	x[ 0 ] = -segmentXLength / 2.0;

	for( size_t xp = 0; xp < image.getWidth(); xp++ )
	{
		FP y = 5.0 * x[ 0 ] * sin( x[ 0 ] );

		function.define( x ) = y;

		size_t yp = ( segmentYLength / 2.0 - y ) / stepY;

		image.writePixel( xp, yp );

		x[ 0 ] += stepX;
	}

	function.makeConvex( 1, 50 );

	for( ScalarFunction::const_iterator iter = function.begin(); iter != function.end(); ++iter )
	{
		FP x = iter->first[ 0 ] + segmentXLength / 2.0;
		FP y = segmentYLength / 2.0 - iter->second;

		image.writePixel( x / stepX, y / stepY, 0xFFFF0000u );
	}

	save_png( "out.png", image );	
}

void test1_ScalarFunction()
{
	ScalarFunction func;
	FILE* file = fopen( "data1", "w" );

	std::vector< FP > x( 1 );
	x[ 0 ] = -M_PI;

	const size_t stepNumber = 100;
	FP stepX = 2.0 * M_PI / ( FP )stepNumber;

	for( size_t i = 0; i <= stepNumber; i++ )
	{
		FP y = -x[ 0 ] * x[ 0 ] + 5.0;//5.0 * x[ 0 ] * sin( x[ 0 ] );

		func.define( x ) = y;

		x[ 0 ] += stepX;

		fprintf( file, "%lg %lg\n", x[ 0 ], y );
	}

	fclose( file );

	func.makeConvex( 1, 100 );

	file = fopen( "data2", "w" );

	for( ScalarFunction::const_iterator iter = func.begin(); iter != func.end(); ++iter )
	{
		fprintf( file, "%lg %lg\n", iter->first[ 0 ], iter->second );	
	}

	fclose( file );
}

void test2_ScalarFunction()
{
	ScalarFunction func;
	FILE* file = fopen( "data3", "w" );

	FP step = 0.5;

	for ( FP x = -5.0; x <= 5.0; x += step ) 
	{
        for ( FP y = -5.0; y <= 5.0; y += step ) 
        {
            FPVector v( 2 );
            v[ 0 ] = x;
            v[ 1 ] = y;

            if (x * x + y * y >= 25) continue;

            FP z = sqrt( x * x + y * y ) * sin( sqrt( x * x + y * y ) );
            func.define( v ) = z;

            fprintf( file, "%g %g %g\n", x, y, z );
        }
    }

	fclose( file );

	func.makeConvex( 2, 10 );

	file = fopen( "data4", "w" );

	for( ScalarFunction::const_iterator iter = func.begin(); iter != func.end(); ++iter )
	{
		fprintf( file, "%g %g %g\n", iter->first[ 0 ], iter->first[ 1 ], iter->second );	
	}

	fclose( file );
}

int main()
{
	test_makeConvexHull();
	test1_ScalarFunction();
	test2_ScalarFunction();

	return 0;
}
