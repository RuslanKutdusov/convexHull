#include <png.h>
#include <stdio.h>
#include <stdlib.h>

//
#include "Image.hpp"


//
void PNGAPI error_function( png_structp png, png_const_charp dummy ) 
{
	( void )dummy; 
	longjmp( png_jmpbuf( png ), 1 );
}


//
int save_png( const char* file_name, const Image& image )
{
	uint8_t* rgb = new uint8_t[ image.getSize() * 3 ];

	int i = 0;
 	for( int j = 0; j < image.getSize(); j++ )
	{
		rgb[ i++ ] = RED( image.pointer()[ j ] );
		rgb[ i++ ] = GREEN( image.pointer()[ j ] );
		rgb[ i++ ] = BLUE( image.pointer()[ j ] );
	}

	png_structp png;
	png_infop info;
	png_uint_32 y;
	png = png_create_write_struct( PNG_LIBPNG_VER_STRING, NULL, NULL, NULL );
	if( png == NULL )
	{
		delete[] rgb;
		return -1;
	}

	info = png_create_info_struct( png );
	if( info == NULL )
	{
		delete[] rgb;
		png_destroy_write_struct( &png, NULL );
		return -1;
	}

	if( setjmp( png_jmpbuf( png ) ) )
	{
		delete[] rgb;
		png_destroy_write_struct( &png, &info );
		return -1;
	}

	FILE * fp = NULL;
	fp = fopen( file_name, "wb" );
	if( fp == NULL )
	{
		delete[] rgb;
		png_destroy_write_struct( &png, &info );
		return -1;
	}

	png_init_io( png, fp );
	png_set_IHDR( png, info, image.getWidth(), image.getHeight(), 8,
		   PNG_COLOR_TYPE_RGB,
		   PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
		   PNG_FILTER_TYPE_DEFAULT );
	png_write_info( png, info );

	for ( y = 0; y < image.getHeight(); ++y )
	{
		png_bytep row = rgb + y * image.getWidth() * 3;
		png_write_rows( png, &row, 1 );
	}

	png_write_end( png, info );

	png_destroy_write_struct( &png, &info );

	fclose( fp );

	return 0;
}


//
uint8_t ALPHA( uint32_t argb )
{
	return *( ( uint8_t* )( &argb ) + 3 );
}


//
uint8_t RED( uint32_t argb )
{
	return *( ( uint8_t* )( &argb ) + 2 );
}


//
uint8_t GREEN( uint32_t argb )
{
	return *( ( uint8_t* )( &argb ) + 1 );
}


//
uint8_t BLUE( uint32_t argb )
{
	return *( ( uint8_t* )( &argb ) );
}
