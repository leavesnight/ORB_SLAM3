//
// Created by leavesnight on 9/14/23.
//

#pragma once

#include "nav_state.h"
#include "common/type_def.h"
#include "common/eigen/eigen_utils.h"

namespace VIEO_SLAM {

template <class Tdata, class Tcalc = double>
class NavStateOdom : public NavState<Tdata, Tcalc> {
  using Base = NavState<Tdata, Tcalc>;

 public:
  using typename Base::Vec3data;

  // b_acc_omega_bb
  Vec3data a_w_ = Vec3data::Zero();
  // w_acc_wb(linear)
  Vec3data a_l_ = Vec3data::Zero();

  // for Eigen has deep copy, we don't define default copy constructor and operator= and we don't need ~NavState(),
  // so move copy constructor and move operator= will also be default

  std::string FormatString(bool bfull = true) const override {
    return Base::FormatString(bfull) + mlog::FormatString(";aw={};al={}", a_w_.transpose(), a_l_.transpose());
  }
};

using NavStateOdomd = NavStateOdom<double>;

class LowFreqPose {
 public:
  template <typename _Tp>
  using vector = std::vector<_Tp>;
  template <typename _Tp>
  using aligned_vector = Eigen::aligned_vector<_Tp>;

  // vector designed for future slave/master timestamps
  // though for LowFreqPose a_w/l is not necessary now, but for easier implementation, we choose NavStateOdom
  aligned_vector<NavStateOdomd> vns_ = aligned_vector<NavStateOdomd>(1);

  typedef enum _TrackState : uint8_t {
    kTrackInvalid = 0,           // Orientation and Position invalid
    kTrackOrientation = 1 << 0,  // Orientation is currently tracked
    kTrackPosition = 1 << 1,     // Position is currently tracked
    kTrackStill = 1 << 2,        // output to system slam is still
    kTrackNavStateBRB = 1 << 3,  // output to system ns_brb_part
    kTrackAligned = 1 << 4,      // Aligned with other slams
    kTrackDerived6DOF = 1 << 5,  // Derived 6Dof by image + odoms
    kTrackStillRelated = (0xFF - kTrackStill - kTrackDerived6DOF)
  } TrackState;
  uint8_t track_state_ = kTrackInvalid;

  using Ttime = common::TimeStamp;
  vector<Ttime> vtimestamp_ = vector<Ttime>(1, INVALID_TIMESTAMP);
};
class HighFreqPose : public LowFreqPose {
 public:
  HighFreqPose(int device_id = -1) : device_id_(device_id) {}

  int device_id_ = 0;
};

}  // namespace VIEO_SLAM
