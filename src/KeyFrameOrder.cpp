//
// Created by leavesnight on 2024/5/3.
//

#include "KeyFrameOrder.h"
#include "KeyFrame.h"

namespace ORB_SLAM3 {

bool KFIdCompare::operator()(const KeyFrame *kfleft, const KeyFrame *kfright) const {
  return kfleft->mnId < kfright->mnId;
}

}  // namespace ORB_SLAM3
