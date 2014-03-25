#pragma once
#include <vector>
#include <float.h>
#include <stdint.h>

#include "FunctionOfAny.hpp"

typedef double FP;
typedef std::vector< FP > FPVector;

#define EPSILON FLT_MIN
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
