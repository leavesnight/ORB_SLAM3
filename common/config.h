//
// Created by leavesnight on 2021/12/22.
//

#pragma once

#include <array>
#include <cmath>

namespace VIEO_SLAM {
// Print related params
//#define TIMER_FLOW
//#define CHECK_NUMERICAL_SAFETY

// Strategy related params
//#define USE_STRATEGY_ABANDON
#ifndef USE_STRATEGY_ABANDON
#define USE_STRATEGY_MIN_DIST
#endif

//#define CHECK_REPLACE_ALL
//#define USE_SIMPLE_REPLACE

// Draw related params
//#define DRAW_ALL_KPS
//#define DRAW_KP2MP_LINE

// IMU related params
#define USE_IMU_CALIBRATE
//#define USE_ACC_AVG_IN_WORLD
#ifndef USE_ACC_AVG_IN_WORLD
#define USE_CPLPI
#ifndef USE_CPLPI
#define USE_RK4_IMU_PREINT
#endif
#else
#define USE_CPLPI  // Global CPLPI
//#define USE_VINSMONO_PREINT
#endif
#define USE_FULL_J_COV
// relative imu factor, now unfinished through GetGravityVec()
//#define NO_GRAVITY

// Strategy related params
// BA with IMU related params
constexpr double kRatioIMUSigma = 1e4;  // 1e3 / 9;
// g then a
constexpr double kCoeffDeltatPrior[2] = {kRatioIMUSigma, kRatioIMUSigma * 1e-3};  // 1e-4};
constexpr double kCoeffIntegrateMidTheorem = 0;                                   // 4e-4 * kRatioIMUSigma;
// Pure Odom related params
constexpr float kThDTPureIMU = 5.f;  // 5 is for TUM_VI slidesX datasets
constexpr float kThDTIMULowFreqStuck = 1.5f;

// PCL related params
constexpr int kNumParamsPointCloud = 4;
constexpr std::array<float, kNumParamsPointCloud> DefaultParamsPointCloud = {0, INFINITY, 2, 0};

}  // namespace VIEO_SLAM
