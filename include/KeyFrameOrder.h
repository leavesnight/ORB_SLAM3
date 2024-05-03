//
// Created by leavesnight on 2024/5/3.
//

#pragma once

namespace ORB_SLAM3 {

class KeyFrame;

// ready for mspKeyFrames set less func., used in IMU Initialization thread, and I think it may help the insert speed
class KFIdCompare {
 public:
  bool operator()(const KeyFrame* kfleft, const KeyFrame* kfright) const;
};

}  // namespace ORB_SLAM3
