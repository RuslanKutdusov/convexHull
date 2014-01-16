#include <stdio.h>
#include <fstream>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include "ScalarFunction.hpp"


//
void defineFunc( const float& step, ScalarFunction& func )
{
	printf( "Define func...\n" );

	for( FP x = -5.0; x <= 5.0; x += step ) 
	{
		for ( FP y = -5.0; y <= 5.0; y += step ) 
		{
			FPVector v( 2 );
			v[ 0 ] = x;
			v[ 1 ] = y;

			if( x * x + y * y >= 25 ) continue;
			FP z = sqrt( x * x + y * y ) * sin( sqrt( x * x + y * y ) );
			func.define( v ) = z;
		}
	}    
}


//
void saveFunc( ScalarFunction& func, const char* filename )
{
	std::ofstream ofs( filename );

    boost::archive::text_oarchive oa( ofs );
    oa << func;
}


//
int main( int argc, char* argv[] )
{
	float step;
	int normalNumber;
	int threadNumber = 2;

	if( argc < 4 )
	{
		printf( "./time_measurements <step> <normalNumber> <threadNumber>\n" );
		return -1;
	}

	sscanf( argv[ 1 ], "%f", &step );
	sscanf( argv[ 2 ], "%d", &normalNumber );
	sscanf( argv[ 3 ], "%d", &threadNumber );

	//
	ScalarFunction func;

	defineFunc( step, func );
	saveFunc( func, "func.dat" );
	
	printf( "Points number: %lu\n", func.size() );

	timespec tp;
	double startTime, endTime, buildTime;

	//
	printf( "makeConvexMultiThread...\n" );
	clock_gettime( CLOCK_REALTIME, &tp );
	startTime = tp.tv_sec + tp.tv_nsec / 1000000000.0;

	func.makeConvexMultiThread( 2, normalNumber, 2 );

	clock_gettime( CLOCK_REALTIME, &tp );
	endTime = tp.tv_sec + tp.tv_nsec / 1000000000.0;

	buildTime = endTime - startTime;
	printf( "Multi-threaded: %g\n", buildTime );
	saveFunc( func, "hullMT.dat" );

	//
	defineFunc( step, func );

	//
	printf( "makeConvexGPU...\n" );
	clock_gettime( CLOCK_REALTIME, &tp );
	startTime = tp.tv_sec + tp.tv_nsec / 1000000000.0;

	func.makeConvexGPU( 2, normalNumber );
	clock_gettime( CLOCK_REALTIME, &tp );
	endTime = tp.tv_sec + tp.tv_nsec / 1000000000.0;

	buildTime = endTime - startTime;
	printf( "GPU: %g\n", buildTime );	
	saveFunc( func, "hullGPU.dat" );

	return 0;
}