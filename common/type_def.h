//
// Created by leavesnight on 6/9/23.
//

#pragma once

namespace VIEO_SLAM {
namespace common {
#define INVALID_TIMESTAMP (-1)
using TimeStamp = double;                      // uint64_t;
using DeltaTimeStamp = double;                 // int64_t;
constexpr double CoeffTimeStampToSecond = 1.;  // 1.e-9;
constexpr double TS2S(const TimeStamp &ts) {
  return INVALID_TIMESTAMP == ts ? INVALID_TIMESTAMP : static_cast<double>(ts * CoeffTimeStampToSecond);
}
constexpr double DTS2S(const DeltaTimeStamp &dts) {
  return INVALID_TIMESTAMP == dts ? INVALID_TIMESTAMP : static_cast<double>(dts * CoeffTimeStampToSecond);
}
constexpr TimeStamp S2TS(const double &t) {
  return INVALID_TIMESTAMP == t ? INVALID_TIMESTAMP : static_cast<TimeStamp>(t / CoeffTimeStampToSecond);
}
constexpr DeltaTimeStamp S2DTS(const double &dt) {
  return INVALID_TIMESTAMP == dt ? INVALID_TIMESTAMP : static_cast<DeltaTimeStamp>(dt / CoeffTimeStampToSecond);
}

using FrameId = unsigned long;
static constexpr FrameId InvalidFrameId = (FrameId)(-1);

}  // namespace common

using FLT_VIEO = double;
using FLT_CAMM = float;

}  // namespace VIEO_SLAM
