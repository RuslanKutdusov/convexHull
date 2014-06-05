// visualize convex hull
// control:
// left/right arrow - rotate around Z axis
// up/down arrow - up/down eye pos
// page up/page down - up/down look at 
// F - show/hide function
// H - show/hide convex hull
// W/S -
// LSHIFT - force
#include "GL/glew.h"
#include <GL/gl.h>
#include <GL/glu.h>
#include <SDL/SDL.h>
#include <time.h>
#include <unistd.h>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <fstream>

#include "ScalarFunction.hpp"

#define WINDOW_WIDTH  640
#define WINDOW_HEIGHT 480


bool key[321];

//
ScalarFunction g_func;
ScalarFunction g_convexHull;
FP g_funcMinVal = FLT_MAX;
FP g_funcMaxVal = -FLT_MAX;
FP g_convexHullMinVal = FLT_MAX;
FP g_convexHullMaxVal = -FLT_MAX;

//
size_t g_funcVertexNumber;
GLuint g_funcVBO;
size_t g_convexVertexNumber;
GLuint g_convexHullVBO;

//
struct Vertex
{
   float x;
   float y;
   float z;
   Vertex(){}
   Vertex( float x_, float y_, float z_ )
      : x( x_ ), y( y_ ), z( z_ )
   {

   }
   void operator()( float x_, float y_, float z_ )
   {
      x = x_;
      y = y_;
      z = z_;
   }
};


//
bool events()
{
   SDL_Event event;
   if( SDL_PollEvent(&event) )
   {
      switch( event.type )
      {
         case SDL_KEYDOWN : key[ event.key.keysym.sym ]=true ;   break;
         case SDL_KEYUP   : key[ event.key.keysym.sym ]=false;   break;
         case SDL_QUIT    : return false; break;
      }
   }
   return true;
}


//
void main_loop_function()
{
   float angleY = 0.0f;
   float angleZ = 0.0f;
   const float angleStep = 0.01f;
   float camZ = -20.0f;
   const float camZStep = 0.1f;
   float camAtZ = 0.0f;

   bool drawFunc = true;
   bool lastKeyFState = key[ SDLK_f ];
   bool drawConvexHull = true;
   bool lastKeyHState = key[ SDLK_h ];

   while( events() )
   {
      glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
      glLoadIdentity();

      gluLookAt( sin( angleZ ) * cos( angleY ) * camZ, cos( angleZ ) * cos( angleY ) * camZ, sin( angleY ) * camZ + camAtZ,
               0.0f, 0.0f, camAtZ, 
               0.0f, 0.0f, 1.0f );

      if( drawFunc )
      {
         glBindBuffer( GL_ARRAY_BUFFER, g_funcVBO );

         glEnableClientState( GL_VERTEX_ARRAY );
         glEnableClientState( GL_COLOR_ARRAY );

         glVertexPointer( 3, GL_FLOAT, 0, 0 );
         glColorPointer( 3, GL_FLOAT, 0, (void*)( sizeof( Vertex ) * g_funcVertexNumber ) );
         glDrawArrays( GL_POINTS, 0, g_funcVertexNumber );  

         glDisableClientState( GL_VERTEX_ARRAY );
         glDisableClientState( GL_COLOR_ARRAY );
      }

      if( drawConvexHull )
      {
         glBindBuffer( GL_ARRAY_BUFFER, g_convexHullVBO );

         glEnableClientState( GL_VERTEX_ARRAY );
         glEnableClientState( GL_COLOR_ARRAY );

         glVertexPointer( 3, GL_FLOAT, 0, 0 );
         glColorPointer( 3, GL_FLOAT, 0, (void*)( sizeof( Vertex ) * g_convexVertexNumber ) );
         glDrawArrays( GL_POINTS, 0, g_convexVertexNumber );  
         
         glDisableClientState( GL_VERTEX_ARRAY );
         glDisableClientState( GL_COLOR_ARRAY );
      }

      glBegin( GL_LINES );
         glColor3ub( 255, 0, 0 ); glVertex3f( 0.0f, 0.0f, 0.0f );
         glColor3ub( 255, 0, 0 ); glVertex3f( 1000.0f, 0.0f, 0.0f );

         glColor3ub( 0, 255, 0 ); glVertex3f( 0.0f, 0.0f, 0.0f );
         glColor3ub( 0, 255, 0 ); glVertex3f( 0.0f, 1000.0f, 0.0f );

         glColor3ub( 0, 0, 255 ); glVertex3f( 0.0f, 0.0f, 0.0f );
         glColor3ub( 0, 0, 255 ); glVertex3f( 0.0f, 0.0f, 1000.0f );
      glEnd();

   	SDL_GL_SwapBuffers();

      float force = 1.0f;

      if( key[ SDLK_LSHIFT ] )
         force = 5.0f;
      
      if( key[ SDLK_RIGHT ] ) 
         angleZ -= angleStep * force; 
      if( key[ SDLK_LEFT ] )
         angleZ += angleStep * force; 

      if( key[ SDLK_UP ] ) 
      {
         angleY -= angleStep * force; 
         if( angleY <= -M_PI_2 )
            angleY = -M_PI_2;
      }
      if( key[ SDLK_DOWN ] )
      {
         angleY += angleStep * force; 
         if( angleY >= M_PI_2 )
            angleY = M_PI_2;
      }

      if( key[ SDLK_PAGEUP ] ) 
         camAtZ += camZStep * force;
      if( key[ SDLK_PAGEDOWN ] )
         camAtZ -= camZStep * force;

      if( key[ SDLK_w ] ) 
         camZ += camZStep * force;
      if( key[ SDLK_s ] )
         camZ -= camZStep * force;

      if( key[ SDLK_f ] != lastKeyFState && !key[ SDLK_f ] )
         drawFunc = !drawFunc;
      lastKeyFState = key[ SDLK_f ];

      if( key[ SDLK_h ] != lastKeyHState && !key[ SDLK_h ] )
         drawConvexHull = !drawConvexHull;
      lastKeyHState = key[ SDLK_h ];

      if( key[ SDLK_ESCAPE ] )
         break;
   }
}


//
void GL_Setup(int width, int height)
{
   glViewport( 0, 0, width, height );

   glMatrixMode( GL_PROJECTION );
   glEnable( GL_DEPTH_TEST );
   glLoadIdentity();
   gluPerspective( 60.0f, (float)width/height, 0.01f, 100000.0f );

   glMatrixMode( GL_MODELVIEW );
}


//
void readFunc( const char* filename, ScalarFunction& func )
{
   std::ifstream ifs( filename );
   boost::archive::text_iarchive ia( ifs );
   ia >> func;
}


//
void readFuncs( const char* func, const char* hull )
{
   readFunc( func, g_func );
   readFunc( hull, g_convexHull );

   for( ScalarFunction::const_iterator iter = g_func.begin(); iter != g_func.end(); ++iter )
   {
      FP z = iter->second;
      if( z > g_funcMaxVal )
         g_funcMaxVal = z;
      if( z < g_funcMinVal )
         g_funcMinVal = z;
   }

   for( ScalarFunction::const_iterator iter = g_convexHull.begin(); iter != g_convexHull.end(); ++iter )
   {
      FP z = iter->second;
      if( z > g_convexHullMaxVal )
         g_convexHullMaxVal = z;
      if( z < g_convexHullMinVal )
         g_convexHullMinVal = z;
   }
}


//
void GenBuffer( GLuint* vbo, size_t size, Vertex* vertex, Vertex* color )
{
   glGenBuffers( 1, vbo );

   glBindBuffer( GL_ARRAY_BUFFER, *vbo );

   size_t length = sizeof( Vertex ) * size;
   glBufferData( GL_ARRAY_BUFFER, length * 2, NULL, GL_STATIC_DRAW );
   glBufferSubData( GL_ARRAY_BUFFER, 0, length, vertex );
   glBufferSubData( GL_ARRAY_BUFFER, length, length, color );

   glBindBuffer( GL_ARRAY_BUFFER, 0 );
}


//
void CreateBuffers()
{
   g_funcVertexNumber = g_func.size();
   g_convexVertexNumber = g_convexHull.size();

   Vertex* funcVertexData = new Vertex[ g_funcVertexNumber ];
   Vertex* funcColorData = new Vertex[ g_funcVertexNumber ];
   Vertex* convexHullVertexData = new Vertex[ g_convexVertexNumber ];
   Vertex* convexHullColorData = new Vertex[ g_convexVertexNumber ];

   size_t i = 0;
   for( ScalarFunction::const_iterator iter = g_func.begin(); iter != g_func.end(); ++iter )
   {
      funcVertexData[ i ]( iter->first[ 0 ], iter->first[ 1 ], iter->second );

      float colorFactor = ( iter->second - g_funcMinVal ) / ( g_funcMaxVal - g_funcMinVal );
      funcColorData[ i ]( colorFactor, colorFactor * 0.5f, 1.0f - colorFactor );      

      i++;
   }

   i = 0;
   for( ScalarFunction::const_iterator iter = g_convexHull.begin(); iter != g_convexHull.end(); ++iter )
   {
      convexHullVertexData[ i ]( iter->first[ 0 ], iter->first[ 1 ], iter->second );

      float colorFactor = ( iter->second - g_convexHullMinVal ) / ( g_convexHullMaxVal - g_convexHullMinVal );
      convexHullColorData[ i ]( colorFactor, 1.0f - colorFactor, 0.0f );      

      i++;
   }   

   GenBuffer( &g_funcVBO, g_funcVertexNumber, funcVertexData, funcColorData );
   GenBuffer( &g_convexHullVBO, g_convexVertexNumber, convexHullVertexData, convexHullColorData );

   delete[] funcVertexData;
   delete[] funcColorData;
   delete[] convexHullVertexData;
   delete[] convexHullColorData;
}


//
int main( int argc, char* argv[] )
{
   int windowWidth = WINDOW_WIDTH;
   int windowHeight = WINDOW_HEIGHT;
   int fullscreen = 0;

   if( argc < 3 )
   {
      printf( "./vis <serialized table func filename> <serialized table convex hull filename> [ <windowWidth> <windowHeight> ]\n" );
      printf( "./vis <serialized table func filename> <serialized table convex hull filename> [ <windowWidth> <windowHeight> <fullscreen = 1, 0 > ]\n" );
      return -1;
   }

   readFuncs( argv[ 1 ], argv[ 2 ] );

   if( argc >= 5 )
   {
      sscanf( argv[ 3 ], "%d", &windowWidth );
      sscanf( argv[ 4 ], "%d", &windowHeight );
   }
   if( argc >= 6 )
      sscanf( argv[ 5 ], "%d", &fullscreen );


   SDL_Init( SDL_INIT_VIDEO );
   const SDL_VideoInfo* info = SDL_GetVideoInfo();	

   int vidFlags = SDL_OPENGL | SDL_GL_DOUBLEBUFFER;
   if( fullscreen )
      vidFlags |= SDL_FULLSCREEN;

   if( info->hw_available )
      vidFlags |= SDL_HWSURFACE;
   else 
      vidFlags |= SDL_SWSURFACE;

   int bpp = info->vfmt->BitsPerPixel;
   SDL_SetVideoMode( windowWidth, windowHeight, bpp, vidFlags );

   GL_Setup( windowWidth, windowHeight );

   GLenum err = glewInit();
   if (GLEW_OK != err)
      fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
   fprintf(stdout, "Status: Using GLEW %s\n", glewGetString(GLEW_VERSION));

   CreateBuffers();

   main_loop_function();

   glDeleteBuffers(1, &g_funcVBO );
   glDeleteBuffers(1, &g_convexHullVBO );
}
