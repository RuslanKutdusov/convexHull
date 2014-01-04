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

GLuint g_funcVBO_V;
GLuint g_funcVBO_C;
GLuint g_convexHullVBO_V;
GLuint g_convexHullVBO_C;

struct Vertex
{
   float x;
   float y;
   float z;
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
         for( ScalarFunction::const_iterator iter = g_func.begin(); iter != g_func.end(); ++iter )
         {
            glBegin( GL_POINTS );

            if( !drawConvexHull )
            {
               float colorFactor = ( iter->second - g_funcMinVal ) / ( g_funcMaxVal - g_funcMinVal );
               glColor3ub( 255 * colorFactor, 127 * colorFactor, 255 * ( 1.0f - colorFactor ) );
            }
            else
               glColor3ub( 255, 255, 0 );

            glVertex3f( iter->first[ 0 ], iter->first[ 1 ], iter->second );

            glEnd();
         }
      }

      if( drawConvexHull )
         for( ScalarFunction::const_iterator iter = g_convexHull.begin(); iter != g_convexHull.end(); ++iter )
         {
            glBegin( GL_POINTS );
            
            if( !drawFunc )
            {
               float colorFactor = ( iter->second - g_convexHullMinVal ) / ( g_convexHullMaxVal - g_convexHullMinVal );
               glColor3ub( 255 * colorFactor, 127 * colorFactor, 255 * ( 1.0f - colorFactor ) );
            }
            else
               glColor3ub( 0, 255, 255 );

            glVertex3f( iter->first[ 0 ], iter->first[ 1 ], iter->second );

            glEnd();
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
   FP step = 0.05f;

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

   g_convexHull.makeConvex( 2, 10 );

   for( ScalarFunction::const_iterator iter = g_convexHull.begin(); iter != g_convexHull.end(); ++iter )
   {
      FP z = iter->second;
      if( z > g_convexHullMaxVal )
         g_convexHullMaxVal = z;
      if( z < g_convexHullMinVal )
         g_convexHullMinVal = z;
   }
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

   buildConvex();

   main_loop_function();
}
