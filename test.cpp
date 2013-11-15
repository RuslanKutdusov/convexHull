#include <math.h>
#include <stdio.h>

#include "Image.hpp"
#include "FunctionOfAny.hpp"
#include "convexHull.hpp"

#define PI 3.1415926f

void testImage()
{
	float segmentXLength = 2.0f * PI + PI;
	float segmentYLength = 4.0f;

	Image image( 512, 512 );
	image.clearByWhite();

	float stepX = segmentXLength / image.getWidth();
	float stepY = segmentYLength / image.getHeight();

	float x = -PI;
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

	float dFi = PI / 10.0f;

	for( float fi = 0.0f; fi < PI; fi += dFi )
	{
		float x = cos( fi )  + segmentXLength / 2.0f;
		float y = segmentYLength / 2.0f - sin( fi );

		image.writePixel( x / stepX, y / stepY );
	}

	save_png( "out.png", image );
}

void test_makeConvexHull()
{
	float segmentXLength = 2.0 * PI;
	float segmentYLength = 20.0f;

	Image image( 512, 512 );
	image.clearByWhite();

	float stepX = segmentXLength / image.getWidth();
	float stepY = segmentYLength / image.getHeight();

	FunctionOfAny< std::vector< FP >, FP > function;

	std::vector< FP > x( 1 );
	x[ 0 ] = -segmentXLength / 2.0;

	for( size_t xp = 0; xp < image.getWidth(); xp++ )
	{
		FP y = 5.0 * x[ 0 ] * sin( x[ 0 ] );

		function.define( x ) = y;

		size_t yp = ( segmentYLength / 2.0 - y ) / stepY;

		image.writePixel( xp, yp );

		x[ 0 ] += stepX;
	}

	makeConvexHull( function, 50 );

	for( FunctionOfAny< std::vector< FP >, FP >::const_iterator iter = function.begin(); iter != function.end(); ++iter )
	{
		FP x = iter->first[ 0 ] + segmentXLength / 2.0;
		FP y = segmentYLength / 2.0 - iter->second;

		image.writePixel( x / stepX, y / stepY, 0xFFFF0000u );
	}

	save_png( "out.png", image );	
}

int main()
{
	test_makeConvexHull();
	return 0;
}
