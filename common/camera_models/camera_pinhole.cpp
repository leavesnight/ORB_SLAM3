//
// Created by leavesnight on 6/6/23.
//

#include <opencv2/core/core.hpp>
#include "camera_pinhole.h"
#include "common/params.h"
#include "include/Converter.h"

namespace VIEO_SLAM {
namespace camm {
PinholeCamera::PinholeCamera(cv::FileStorage &fs_settings, int id, bool &bmiss_param)
    : Base(id, 0, 0, vector<Tdata>(), SE3data()) {
  string cam_name = "Camera" + ParamsStorage::GetInputIdStr(id);

  cv::FileNode node_tmp = fs_settings[cam_name + ".fx"];
  if (node_tmp.empty()) {
    bmiss_param = true;
    return;
  }
  float fx = node_tmp;
  node_tmp = fs_settings[cam_name + ".fy"];
  if (node_tmp.empty()) {
    bmiss_param = true;
    return;
  }
  float fy = node_tmp;
  node_tmp = fs_settings[cam_name + ".cx"];
  if (node_tmp.empty()) {
    bmiss_param = true;
    return;
  }
  float cx = node_tmp;
  node_tmp = fs_settings[cam_name + ".cy"];
  if (node_tmp.empty()) {
    bmiss_param = true;
    return;
  }
  float cy = node_tmp;
  node_tmp = fs_settings[cam_name + ".imageScale"];
  if (!node_tmp.empty()) {
    image_scale_ = (Tdata)node_tmp;
  }

  parameters_ = {fx, fy, cx, cy};
  if (image_scale_ != (Tdata)(1.f)) {
    for (int i = 0; i < 4; ++i) parameters_[i] *= image_scale_;
    width_ = (Tsize)(width_ * image_scale_);
    height_ = (Tsize)(height_ * image_scale_);
  }

  using std::endl;
  PRINT_INFO_MUTEX(endl << "Camera (Pinhole) Parameters: image_scale=" << image_scale_ << endl);
  PRINT_INFO_MUTEX("- fx: " << fx << endl);
  PRINT_INFO_MUTEX("- fy: " << fy << endl);
  PRINT_INFO_MUTEX("- cx: " << cx << endl);
  PRINT_INFO_MUTEX("- cy: " << cy << endl);

  node_tmp = fs_settings[cam_name + ".Trc"];
  if (!node_tmp.empty()) {
    cv::Mat cvTrc = node_tmp.mat();
    if (cvTrc.rows != 3 || cvTrc.cols != 4) {
      std::cerr << "*Trc matrix have to be a 3x4 transformation matrix*" << std::endl;
      bmiss_param = true;
      return;
    }
    SetTrc(SE3data(Sophus::SO3ex<TdataGeo>(ORB_SLAM3::Converter::toMatrix3d(cvTrc.rowRange(0, 3).colRange(0, 3)).cast<TdataGeo>()),
                   ORB_SLAM3::Converter::toVector3d(cvTrc.col(3)).cast<TdataGeo>()));
  } else {
    PRINT_INFO_MUTEX("Warning:*Trc matrix doesn't exist*" << std::endl);
  }
  PRINT_INFO_MUTEX("- Trc: \n" << Trc_.matrix3x4() << std::endl);

  bmiss_param = false;

  camera_model_ = CameraModel::kPinhole;
}
bool PinholeCamera::ParseCamParamFile(cv::FileStorage &fs_settings, int id, Camera::Ptr &pCamInst) {
  bool b_miss_params = false;
  pCamInst = std::make_shared<PinholeCamera>(fs_settings, id, b_miss_params);
  using std::cerr;
  using std::endl;
  if (b_miss_params) {
    cerr << "Error: miss params!" << endl;
    return false;
  }

  return true;
}

Eigen::Vector2d PinholeCamera::Project(const Vec3io &p_3d) const {
  Eigen::Vector2d res;
  res[0] = parameters_[0] * p_3d[0] / p_3d[2] + parameters_[2];
  res[1] = parameters_[1] * p_3d[1] / p_3d[2] + parameters_[3];

  return res;
}

}  // namespace camm
}  // namespace VIEO_SLAM