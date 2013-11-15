#include <math.h>

#include "Image.hpp"

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
		float y = sin( x ) + segmentYLength / 2.0f;

		size_t yp = y / stepY;

		image.writePixel( xp, yp );

		x += stepX;
	}

	save_png( "out.png", image );
}

int main()
{
	testImage();

	return 0;
}
