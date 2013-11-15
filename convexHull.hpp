#pragma once
#include <vector>
#include "FunctionOfAny.hpp"

typedef double FP;


//
void makeConvexHull( FunctionOfAny< std::vector< FP >, FP >& function, const int32_t& stepNumber );
