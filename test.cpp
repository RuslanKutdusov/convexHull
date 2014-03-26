#include <math.h>
#include <stdio.h>

#include "ScalarFunction.hpp"


//
void test1_ScalarFunction()
{
	ScalarFunction func;
	FILE* file = fopen( "plots/points_test1_func.plot", "w" );

	std::vector< FP > x( 1 );
	x[ 0 ] = -M_PI;

	const size_t stepNumber = 100;
	FP stepX = 2.0 * M_PI / ( FP )stepNumber;

	for( size_t i = 0; i <= stepNumber; i++ )
	{
		FP y = 5.0 * x[ 0 ] * sin( x[ 0 ] );

		func.define( x ) = y;

		x[ 0 ] += stepX;

		fprintf( file, "%g %g\n", x[ 0 ], y );
	}

	fclose( file );

	func.makeConvex( 1, 100 );

	file = fopen( "plots/points_test1_ch.plot", "w" );

	for( ScalarFunction::const_iterator iter = func.begin(); iter != func.end(); ++iter )
	{
		fprintf( file, "%g %g\n", iter->first[ 0 ], iter->second );	
	}

	fclose( file );
}


//
void test1_ScalarFunctionMT()
{
	ScalarFunction func;
	FILE* file = fopen( "plots/points_test1MT_func.plot", "w" );

	std::vector< FP > x( 1 );
	x[ 0 ] = -M_PI;

	const size_t stepNumber = 100;
	FP stepX = 2.0 * M_PI / ( FP )stepNumber;

	for( size_t i = 0; i <= stepNumber; i++ )
	{
		FP y = 5.0 * x[ 0 ] * sin( x[ 0 ] );

		func.define( x ) = y;

		x[ 0 ] += stepX;

		fprintf( file, "%g %g\n", x[ 0 ], y );
	}

	fclose( file );

	func.makeConvexMultiThread( 1, 100, 2 );

	file = fopen( "plots/points_test1MT_ch.plot", "w" );

	for( ScalarFunction::const_iterator iter = func.begin(); iter != func.end(); ++iter )
	{
		fprintf( file, "%g %g\n", iter->first[ 0 ], iter->second );	
	}

	fclose( file );
}


//
void test2_ScalarFunction()
{
	ScalarFunction func;
	FILE* file = fopen( "plots/points_test2_func.plot", "w" );

	FP step = 0.05;

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

	func.makeConvex( 2, 50 );

	file = fopen( "plots/points_test2_ch.plot", "w" );

	for( ScalarFunction::const_iterator iter = func.begin(); iter != func.end(); ++iter )
	{
		fprintf( file, "%g %g %g\n", iter->first[ 0 ], iter->first[ 1 ], iter->second );	
	}

	fclose( file );
}


//
void test2_ScalarFunctionMT()
{
	ScalarFunction func;
	FILE* file = fopen( "plots/points_test2MT_func.plot", "w" );

	FP step = 0.05;

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

	func.makeConvexMultiThread( 2, 50, 2 );

	file = fopen( "plots/points_test2MT_ch.plot", "w" );

	for( ScalarFunction::const_iterator iter = func.begin(); iter != func.end(); ++iter )
	{
		fprintf( file, "%g %g %g\n", iter->first[ 0 ], iter->first[ 1 ], iter->second );	
	}

	fclose( file );
}


//
void test2_ScalarFunctionGPU()
{
	ScalarFunction func;
	FILE* file = fopen( "plots/points_test2GPU_func.plot", "w" );

	FP step = 0.05;

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

	func.makeConvexGPU( 2, 50 );

	file = fopen( "plots/points_test2GPU_ch.plot", "w" );

	for( ScalarFunction::const_iterator iter = func.begin(); iter != func.end(); ++iter )
	{
		fprintf( file, "%g %g %g\n", iter->first[ 0 ], iter->first[ 1 ], iter->second );	
	}

	fclose( file );
}

int main()
{
	test1_ScalarFunction();
	test1_ScalarFunctionMT();
	test2_ScalarFunction();
	test2_ScalarFunctionMT();
	test2_ScalarFunctionGPU();

	return 0;
}
