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

//
typedef std::vector< FP > FPVector;

//
#define PI ( FP )M_PI

//
static const int MAX_GPU_COUNT = 8;


//
class ScalarFunction : public FunctionOfAny< FPVector, FP >
{
public:
	//
	void makeConvex( const uint32_t& dimX, const uint32_t& numberOfPoints );
	//
	void makeConvexMultiThread( const uint32_t& dimX, const uint32_t& numberOfPoints, const uint32_t& jobs );

#ifdef GPU
	void makeConvexGPU( const int& dimX, const int& numberOfPoints );
#endif

private:
#ifdef GPU	
	int 		hyperplanesArraySize;
	FP* 		hyperplanes;
	int 		pointsArraySize;
	FP* 		points;
	FP* 		d_hyperplanes[ MAX_GPU_COUNT ];
	FP* 		d_points[ MAX_GPU_COUNT ];

	int 	 	pointsChunksNumber;
	int 	 	pointsChunksPerDevice;
	int 	 	pointsChunksForLastDevice;

	// особенность суперкомпьютера Уран, 8 видеокарт одного узла по сути разбиты на 2 части
	// такие, что для видеокарт одной части возможен peer access, но для видеокарт из разных частей - нет.
	// определяем для каких видеокарт возможен peer access( они разбиваются на части(группы) ). 
	// выбираем ту группу для работы, кол-во видеокарт в которой больше, чем в другой.
	std::vector< int > devicesGroups[ 2 ];

	//
	void 		CopyData( const int& dimX );
	void 		InitHyperplanes( const int& dimX, const int& numberOfHyperplanes, const FP& dFi );
	int 		PrepareDevices();
	void 		DeviceMemoryPreparing( const int& n, const int& deviceCount );
	int 		CalcPointsNumberPerDevice( const int& device, const int& deviceCount );
	void 		Synchronize();
	void 		FirstStage( const int& dimX, const int& numberOfHyperplanes, const int& deviceCount );
	void 		SecondStage( const int& dimX, const int& numberOfHyperplanes );
	void 		ThirdStage( const int& dimX, const int& numberOfHyperplanes, const int& deviceCount );
	void 		GetResult( const int& dimX, const int& deviceCount );
#endif
};
