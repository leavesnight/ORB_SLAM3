//
// Created by leavesnight on 6/6/23.
//

#include <opencv2/core/core.hpp>
#include "common/params.h"
#include "camera_radtan.h"

namespace VIEO_SLAM {
namespace camm {
RadtanCamera::RadtanCamera(const vector<Tdata> &distcoef, cv::FileStorage &fs_settings, int id, bool &bmiss_param)
    : Base(fs_settings, id, bmiss_param) {
  assert(distcoef.size() == 4 || distcoef.size() == 5);
  parameters_.resize(4 + distcoef.size());
  for (size_t i = 0; i < distcoef.size(); ++i) {
    parameters_[4 + i] = distcoef[i];
  }

  auto sz_param = parameters_.size();
  if (8 > sz_param)
    assert(0);
  else
    num_k_ = sz_param - 6;
  camera_model_ = CameraModel::kRadtan;
  this->num_max_iteration_ = 5;
}
bool RadtanCamera::ParseCamParamFile(cv::FileStorage &fs_settings, int id, Camera::Ptr &pCamInst) {
  string cam_name = "Camera" + ParamsStorage::GetInputIdStr(id);
  cv::FileNode node_tmp = fs_settings[cam_name + ".fx"];
  if (node_tmp.empty()) return false;
  bool b_miss_params = false;

  vector<Tdata> distcoef(4);
  for (int i = 0; i < 2; ++i) {
    distcoef[i] = fs_settings[cam_name + ".k" + std::to_string(i + 1)];
  }
  int id_distcoef = 2;
  node_tmp = fs_settings[cam_name + ".k3"];
  if (!node_tmp.empty()) {
    const float k3 = (float)node_tmp;
    if (k3 != 0) {
      distcoef.resize(5);
      distcoef[id_distcoef++] = k3;
    }
  }
  for (int i = 0; i < 2; ++i) {
    distcoef[id_distcoef++] = fs_settings[cam_name + ".p" + std::to_string(i + 1)];
  }

  pCamInst = std::make_shared<RadtanCamera>(distcoef, fs_settings, id, b_miss_params);
  using std::cerr;
  using std::endl;
  if (b_miss_params) {
    cerr << "Error: miss params!" << endl;
    return false;
  }

  PRINT_INFO_MUTEX(endl << "Camera (Radtan) Parameters: " << endl);
  for (int i = 0; i < 2; ++i) {
    PRINT_INFO_MUTEX("- k" + std::to_string(i + 1) + ": " << distcoef[i] << endl);
  }
  id_distcoef = 2;
  if (distcoef.size() == 5) PRINT_INFO_MUTEX("- k3: " << distcoef[id_distcoef++] << endl);
  for (int i = 0; i < 2; ++i) {
    PRINT_INFO_MUTEX("- p" + std::to_string(i + 1) + ": " << distcoef[id_distcoef++] << endl);
  }

  // TODO: check the input
  return true;
}

}  // namespace camm
}  // namespace VIEO_SLAM