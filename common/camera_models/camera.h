//
// Created by leavesnight on 5/30/23.
//

#pragma once

#include <opencv2/core/persistence.hpp>  // for yaml analysis
#include "camera_radtan.h"
#include "camera_kb8.h"

namespace VIEO_SLAM {
namespace camm {
template <typename... Ts>
Camera::Ptr CreateCameraInstance(const Camera::CameraModel &cam_model, Ts... ts) {
  Camera::Ptr pcam = nullptr;
  using std::static_pointer_cast;
  switch (cam_model) {
    case Camera::CameraModel::kPinhole:
      pcam = static_pointer_cast<Camera>(std::make_shared<PinholeCamera>(ts...));
      break;
    case Camera::CameraModel::kRadtan:
      pcam = static_pointer_cast<Camera>(std::make_shared<RadtanCamera>(ts...));
      break;
    case Camera::CameraModel::kKB8:
      pcam = static_pointer_cast<Camera>(std::make_shared<KB8Camera>(ts...));
      break;
    default:
      PRINT_ERR_MUTEX("Unsupported Camera Model in " << __FUNCTION__ << std::endl);
      exit(-1);
  }
  return pcam;
}

void AutoGetCamerasInfo(cv::FileStorage &fs_settings, std::vector<camm::Camera::Ptr> &pcams,
                        std::vector<float> *pdepth_map_factors = nullptr,
                        std::vector<std::array<float, 2>> *pdepth_limits_ = nullptr,
                        std::vector<std::array<float, kNumParamsPointCloud>> *pparams_point_cloud_ = nullptr,
                        std::vector<int> *pids_depth = nullptr, bool *pusedistort = nullptr);

}  // namespace camm
}  // namespace VIEO_SLAM
