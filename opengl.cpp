#include "GL/glew.h"
#include <GL/gl.h>
#include <GL/glu.h>
#include <SDL/SDL.h>

#include "ScalarFunction.hpp"

#define window_width  640
#define window_height 480


bool key[321];

ScalarFunction g_func;
ScalarFunction g_convexHull;
FP g_funcMinVal = FLT_MAX;
FP g_funcMaxVal = -FLT_MAX;
FP g_convexHullMinVal = FLT_MAX;
FP g_convexHullMaxVal = -FLT_MAX;

size_t g_vertexNumber;
GLuint g_funcVBO;
GLuint g_convexHullVBO;

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


void main_loop_function()
{
   float angleX = 0.0f;
   float angleY = 0.0f;
   float angleZ = 0.0f;
   const float angleStep = 0.1f;
   float camZ = -20.0f;
   const float camZStep = 1.0f;

   bool drawFunc = true;
   bool lastKeyFState = key[ SDLK_f ];
   bool drawConvexHull = true;
   bool lastKeyHState = key[ SDLK_h ];

   while( events() )
   {
      glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
      glLoadIdentity();

      gluLookAt( sin( angleZ ) * cos( angleY ) * camZ, cos( angleZ ) * cos( angleY ) * camZ, sin( angleY ) * camZ,
               0.0f, 0.0f, 0.0f, 
               0.0f, 0.0f, 1.0f );

      if( drawFunc )
      {
         glBindBuffer( GL_ARRAY_BUFFER, g_funcVBO );

         glEnableClientState( GL_VERTEX_ARRAY );
         glEnableClientState( GL_COLOR_ARRAY );

         glVertexPointer( 3, GL_FLOAT, 0, 0 );
         glColorPointer( 3, GL_FLOAT, 0, (void*)( sizeof( Vertex ) * g_vertexNumber ) );
         glDrawArrays( GL_POINTS, 0, g_vertexNumber );  

         glDisableClientState( GL_VERTEX_ARRAY );
         glDisableClientState( GL_COLOR_ARRAY );
      }

      if( drawConvexHull )
      {
         glBindBuffer( GL_ARRAY_BUFFER, g_convexHullVBO );

         glEnableClientState( GL_VERTEX_ARRAY );
         glEnableClientState( GL_COLOR_ARRAY );

         glVertexPointer( 3, GL_FLOAT, 0, 0 );
         glColorPointer( 3, GL_FLOAT, 0, (void*)( sizeof( Vertex ) * g_vertexNumber ) );
         glDrawArrays( GL_POINTS, 0, g_vertexNumber );  
         
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
      
      if( key[ SDLK_RIGHT ] ) 
         angleZ -= angleStep; 
      if( key[ SDLK_LEFT ] )
         angleZ += angleStep; 

      if( key[ SDLK_UP ] ) 
         angleY -= angleStep; 
      if( key[ SDLK_DOWN ] )
         angleY += angleStep; 

      if( key[ SDLK_PAGEUP ] ) 
         angleX -= angleStep; 
      if( key[ SDLK_PAGEDOWN ] )
         angleX += angleStep; 

      if( key[ SDLK_w ] ) 
         camZ += camZStep;
      if( key[ SDLK_s ] )
         camZ -= camZStep;

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


void GL_Setup(int width, int height)
{
   glViewport( 0, 0, width, height );

   glMatrixMode( GL_PROJECTION );
   glEnable( GL_DEPTH_TEST );
   glLoadIdentity();
   gluPerspective( 60.0f, (float)width/height, 0.01f, 100.0f );

   glMatrixMode( GL_MODELVIEW );
}


void buildConvex()
{
   FP step = 0.02f;

   for( FP x = -5.0; x <= 5.0; x += step ) 
   {
      for ( FP y = -5.0; y <= 5.0; y += step ) 
      {
         FPVector v( 2 );
         v[ 0 ] = x;
         v[ 1 ] = y;

         if (x * x + y * y >= 25) continue;

         FP z = sqrt( x * x + y * y ) * sin( sqrt( x * x + y * y ) );
         g_func.define( v ) = z;

         if( z > g_funcMaxVal )
            g_funcMaxVal = z;
         if( z < g_funcMinVal )
            g_funcMinVal = z;
      }
   }    

   g_convexHull = g_func;

   g_convexHull.makeConvex( 2, 5 );

   for( ScalarFunction::const_iterator iter = g_convexHull.begin(); iter != g_convexHull.end(); ++iter )
   {
      FP z = iter->second;
      if( z > g_convexHullMaxVal )
         g_convexHullMaxVal = z;
      if( z < g_convexHullMinVal )
         g_convexHullMinVal = z;
   }
}


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


void CreateBuffers()
{
   g_vertexNumber = g_func.size();

   Vertex* funcVertexData = new Vertex[ g_vertexNumber ];
   Vertex* funcColorData = new Vertex[ g_vertexNumber ];
   Vertex* convexHullVertexData = new Vertex[ g_vertexNumber ];
   Vertex* convexHullColorData = new Vertex[ g_vertexNumber ];

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
      convexHullColorData[ i ]( 1.0f - colorFactor, colorFactor, colorFactor * 0.5f );      

      i++;
   }   

   GenBuffer( &g_funcVBO, g_vertexNumber, funcVertexData, funcColorData );
   GenBuffer( &g_convexHullVBO, g_vertexNumber, convexHullVertexData, convexHullColorData );

   delete[] funcVertexData;
   delete[] funcColorData;
   delete[] convexHullVertexData;
   delete[] convexHullColorData;
}


int main()
{
   SDL_Init( SDL_INIT_VIDEO );
   const SDL_VideoInfo* info = SDL_GetVideoInfo();	

   int vidFlags = SDL_OPENGL | SDL_GL_DOUBLEBUFFER;

   if( info->hw_available )
      vidFlags |= SDL_HWSURFACE;
   else 
      vidFlags |= SDL_SWSURFACE;

   int bpp = info->vfmt->BitsPerPixel;
   SDL_SetVideoMode( window_width, window_height, bpp, vidFlags );

   GL_Setup( window_width, window_height );

   GLenum err = glewInit();
   if (GLEW_OK != err)
   {
     /* Problem: glewInit failed, something is seriously wrong. */
     fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
   }
   fprintf(stdout, "Status: Using GLEW %s\n", glewGetString(GLEW_VERSION));

   buildConvex();
   CreateBuffers();

   main_loop_function();

   glDeleteBuffers(1, &g_funcVBO );
   glDeleteBuffers(1, &g_convexHullVBO );
}
