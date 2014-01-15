#pragma once
#include "ScalarFunction.hpp"

namespace gpu
{

void makeConvex( ScalarFunction& func, const size_t& dimX, const size_t& numberOfPoints );

}