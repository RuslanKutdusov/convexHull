#pragma once
#include "ScalarFunction.hpp"

namespace gpu
{

void makeConvex( ScalarFunction& func, const uint32_t& dimX, const uint32_t& numberOfPoints );

}