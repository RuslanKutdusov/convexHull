#pragma once
#include <vector>
#include <float.h>
#include <stdint.h>

#include "FunctionOfAny.hpp"

#if defined( FLOAT_PRECISION )
	typedef float FP;
	#define EPSILON FLT_MIN
#elif defined( DOUBLE_PRECISION )
	typedef double FP;
	#define EPSILON DBL_MIN
#else
	#error "Unspecified precision"
#endif

typedef std::vector< FP > FPVector;

#define PI ( FP )M_PI

class ScalarFunction : public FunctionOfAny< FPVector, FP >
{
public:
	//
	void makeConvex( const uint32_t& dimX, const uint32_t& numberOfPoints );
	//
	void makeConvexMultiThread( const uint32_t& dimX, const uint32_t& numberOfPoints, const uint32_t& jobs );

#ifdef GPU
	void makeConvexGPU( const uint32_t& dimX, const uint32_t& numberOfPoints );
#endif

private:
	
};
