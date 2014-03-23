#pragma once
#include <vector>
#include <float.h>

#include "FunctionOfAny.hpp"

typedef double FP;
typedef std::vector< FP > FPVector;

#define EPSILON FLT_MIN
#define PI ( FP )M_PI

class ScalarFunction : public FunctionOfAny< FPVector, FP >
{
public:
	//
	void makeConvex( const size_t& dimX, const size_t& numberOfPoints );
	//
	void makeConvexMultiThread( const size_t& dimX, const size_t& numberOfPoints, const size_t& jobs );

#ifdef GPU
	void makeConvexGPU( const size_t& dimX, const size_t& numberOfPoints );
#endif

private:
	
};
