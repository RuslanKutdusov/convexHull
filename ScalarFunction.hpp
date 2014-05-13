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
static const uint32_t MAX_GPU_COUNT = 8;


//
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
#ifdef GPU	
	uint64_t	hyperplanesArraySize;
	FP* 		hyperplanes;
	uint64_t 	pointsArraySize;
	FP* 		points;
	FP* 		d_hyperplanes[ MAX_GPU_COUNT ];
	FP* 		d_points[ MAX_GPU_COUNT ];

	uint32_t  	pointsChunksNumber;
	uint32_t  	pointsChunksPerDevice;
	uint32_t  	pointsChunksForLastDevice;

	uint64_t 	start[ MAX_GPU_COUNT ];
	uint64_t 	stop[ MAX_GPU_COUNT ];

	enum LAUNCH_TIME
	{
		LAUNCH_TIME_HTOD = 0,
		LAUNCH_TIME_STAGE1,
		LAUNCH_TIME_STAGE2_FIRST_GROUP,
		LAUNCH_TIME_STAGE2_SECOND_GROUP,
		LAUNCH_TIME_STAGE2,
		LAUNCH_TIME_STAGE3,
		LAUNCH_TIME_DTOH,

		LAUNCH_TIME_COUNT
	};

	float 		launchTime[ LAUNCH_TIME_COUNT ][ MAX_GPU_COUNT ];

	// особенность суперкомпьютера Уран, 8 видеокарт одного узла по сути разбиты на 2 части
	// такие, что для видеокарт одной части возможен peer access, но для видеокарт из разных частей - нет.
	// определяем для каких видеокарт возможен peer access( они разбиваются на части(группы) ). 
	// выбираем ту группу для работы, кол-во видеокарт в которой больше, чем в другой.
	std::vector< uint32_t > devicesGroups[ 2 ];

	//
	void 		CopyData( const uint32_t& dimX );
	void 		InitHyperplanes( const uint32_t& dimX, const uint32_t& numberOfHyperplanes, const FP& dFi );
	uint32_t 	PrepareDevices( const uint32_t& neededDeviceNumber );
	void 		DeviceMemoryPreparing( const uint32_t& n, const uint32_t& deviceCount );
	uint32_t 	CalcPointsNumberPerDevice( const uint32_t& device, const uint32_t& deviceCount );
	void 		Synchronize( LAUNCH_TIME lt );
	void 		FixLaunchTime( LAUNCH_TIME lt, uint32_t device );
	void 		FirstStage( const uint32_t& dimX, const uint32_t& numberOfHyperplanes, const uint32_t& deviceCount );
	void 		SecondStage( const uint32_t& dimX, const uint32_t& numberOfHyperplanes );
	void 		ThirdStage( const uint32_t& dimX, const uint32_t& numberOfHyperplanes, const uint32_t& deviceCount );
	void 		GetResult( const uint32_t& dimX, const uint32_t& deviceCount );
#endif
};
