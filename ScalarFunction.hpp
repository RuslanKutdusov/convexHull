#pragma once
#include <vector>
#include <float.h>
#include <stdint.h>

#include "FunctionOfAny.hpp"

#if defined( FLOAT_PRECISION )
	typedef float FP;
	// пришлось умножить epsilon на 4.0(взято с потолка) из-за вынесения деления(оптимизация) из ThirdStageKernel в SecondStageKernel, т.к иначе из-за нормалей вида (x,y,0.0) в результате получаем NaN
	#define EPSILON ( FLT_MIN * 4.0 ) 
#elif defined( DOUBLE_PRECISION )
	typedef double FP;
	#define EPSILON ( DBL_MIN * 4.0 )
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
	uint64_t	m_hyperplanesArraySize;
	FP* 		m_hyperplanes;
	uint64_t 	m_pointsArraySize;
	FP* 		m_points;
	FP* 		m_hyperplanesDevPtr[ MAX_GPU_COUNT ];
	FP* 		m_pointsDevPtr[ MAX_GPU_COUNT ];

	uint32_t  	m_pointsChunksNumber;
	uint32_t  	m_pointsChunksPerDevice;
	uint32_t  	m_pointsChunksForLastDevice;

	uint64_t 	m_start[ MAX_GPU_COUNT ];
	uint64_t 	m_stop[ MAX_GPU_COUNT ];

	enum LAUNCH_TIME
	{
		LAUNCH_TIME_HTOD = 0,
		LAUNCH_TIME_STAGE1,
		LAUNCH_TIME_STAGE2_FIRST_GROUP,
		LAUNCH_TIME_STAGE2_SECOND_GROUP,
		LAUNCH_TIME_STAGE2,
		LAUNCH_TIME_COPY_HYPERPLANES,
		LAUNCH_TIME_STAGE3,
		LAUNCH_TIME_DTOH,

		LAUNCH_TIME_COUNT
	};

	float 		m_launchTime[ LAUNCH_TIME_COUNT ][ MAX_GPU_COUNT ];

	// особенность суперкомпьютера Уран, 8 видеокарт одного узла по сути разбиты на 2 части
	// такие, что для видеокарт одной части возможен peer access, но для видеокарт из разных частей - нет.
	std::vector< uint32_t > m_devicesGroups[ 2 ];

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
