//
// Created by leavesnight on 8/10/23.
//

#include "common/params.h"
#include "camera.h"

namespace VIEO_SLAM {
namespace camm {

void AutoGetCamerasInfo(cv::FileStorage &fs_settings, std::vector<camm::Camera::Ptr> &pcams,
                        std::vector<float> *pdepth_map_factors, std::vector<std::array<float, 2>> *pdepth_limits_,
                        std::vector<std::array<float, kNumParamsPointCloud>> *pparams_point_cloud_,
                        std::vector<int> *pids_depth, bool *pusedistort) {
  using std::endl;
  using std::string;
  using std::stringstream;
  using std::vector;

  // auto check camera types by iteratively adding 1 now
  cv::FileNode node_tmp = fs_settings["Camera" + ParamsStorage::GetInputIdStr(0) + ".type"];
  vector<string> scamera_types(1, "Pinhole");
  if (!node_tmp.empty()) {
    scamera_types[0] = string(node_tmp);
    int i = 1, di = 1;
    while (1) {
      node_tmp = fs_settings["Camera" + ParamsStorage::GetInputIdStr(i) + ".type"];
      if (node_tmp.empty()) break;
      scamera_types.emplace_back(string(node_tmp));
      i += di;
    }
  }
  int sz_camera_types = scamera_types.size();
  for (int i = 0; i < sz_camera_types; ++i) {
    PRINT_INFO_MUTEX("Camera" + ParamsStorage::GetInputIdStr(i) + ".type=" << scamera_types[i] << endl);
  }

  // i may not be synced with i_pcams
  int i = 0, sz_cam_types = (int)scamera_types.size(), i_depth = 0;
  pcams.clear();
  stringstream sstr_tmp;
  while (1) {
    pcams.emplace_back();
    bool bexist_cam = false;
    const auto &cam_type = scamera_types[i < sz_cam_types ? i : sz_camera_types - 1];
    if (cam_type == "KannalaBrandt8") {
      if (pusedistort) *pusedistort = true;  // TODO: now fisheye only usedistort_ is confirmed
      bexist_cam = camm::KB8Camera::ParseCamParamFile(fs_settings, i, pcams.back());
    } else {
      if (cam_type == "Pinhole") {
        if (pusedistort && *pusedistort) {
          PRINT_INFO_MUTEX("Warning: Pinhole usedistort!" << endl);
        }
        bexist_cam = camm::PinholeCamera::ParseCamParamFile(fs_settings, i, pcams.back());
      } else if (cam_type == "Radtan") {
        bexist_cam = camm::RadtanCamera::ParseCamParamFile(fs_settings, i, pcams.back());
        // if (pusedistort) *pusedistort = false;
      } else if (cam_type == "Depth") {
        auto cam_prefix = "Camera" + ParamsStorage::GetInputIdStr(i);
        node_tmp = fs_settings[cam_prefix + ".DepthMapFactor"];
        if (!node_tmp.empty()) bexist_cam = true;
        if (bexist_cam) {
          if (pdepth_map_factors) {
            if ((int)pdepth_map_factors->size() <= i_depth) pdepth_map_factors->resize(i_depth + 1);
            auto &depth_map_factor = (*pdepth_map_factors)[i_depth] = (float)node_tmp;
            if (fabs(depth_map_factor) < 1e-5)
              depth_map_factor = 1;
            else
              depth_map_factor = 1.0f / depth_map_factor;
            PRINT_INFO_MUTEX(cam_prefix + ".DepthMapFactor=" << depth_map_factor << endl);
            if (pids_depth) pids_depth->push_back(i);

            if (pdepth_limits_) {
              node_tmp = fs_settings[cam_prefix + ".DepthLimits"];
              auto &param_tmp = *pdepth_limits_;
              if (!node_tmp.empty()) {
                if ((int)param_tmp.size() < i_depth) param_tmp.resize(i_depth + 1);
                sstr_tmp << cam_prefix + ".DepthLimits=";
                assert(2 == node_tmp.size());
                for (int k = 0; k < (int)node_tmp.size(); ++k) {
                  param_tmp[i_depth][k] = (float)node_tmp[k];
                  sstr_tmp << param_tmp[i_depth][k] << " ";
                }
                PRINT_INFO_MUTEX(sstr_tmp.str() << endl);
              }
            }
            if (pparams_point_cloud_) {
              node_tmp = fs_settings[cam_prefix + ".ParamsPointCloud"];
              auto &param_tmp = *pparams_point_cloud_;
              if (!node_tmp.empty()) {
                if ((int)param_tmp.size() < i_depth) param_tmp.resize(i_depth + 1, DefaultParamsPointCloud);
                sstr_tmp.str("");
                sstr_tmp << cam_prefix + ".ParamsPointCloud=";
                assert(2 <= node_tmp.size());
                for (int k = 0; k < kNumParamsPointCloud; ++k) {
                  if (k < (int)node_tmp.size()) {
                    param_tmp[i_depth][k] = (float)node_tmp[k];
                  }
                  sstr_tmp << param_tmp[i_depth][k] << " ";
                }
                PRINT_INFO_MUTEX(sstr_tmp.str() << endl);
              }
            }

            ++i_depth;
            assert(i_depth == 1 && "Unimplement > 1 depth camera!");
          }
          pcams.pop_back();  // Depth won't enter camera models
        }
      } else {
        PRINT_ERR_MUTEX("Unsupported Camera Model in " << __FUNCTION__ << endl);
        exit(-1);
      }
    }
    if (!bexist_cam) {
      pcams.pop_back();
      break;
    }
    ++i;
  }
}

}  // namespace camm
}  // namespace VIEO_SLAM