#pragma once
#include <vector>
#include <float.h>

#include "FunctionOfAny.hpp"

typedef double FP;
typedef std::vector< FP > FPVector;

#define EPSILON FLT_MIN

class ScalarFunction : public FunctionOfAny< FPVector, FP >
{
public:
	void makeConvex( const size_t& dimX, const size_t& numberOfPoints );
private:

};
