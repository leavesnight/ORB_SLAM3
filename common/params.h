//
// Created by leavesnight on 8/11/23.
//

#pragma once

#include <opencv2/core/persistence.hpp>
#include "common/mlog/log.h"

namespace VIEO_SLAM {
class ParamsStorage {
 public:
  using string = std::string;
  template <typename _Tp>
  using vector = std::vector<_Tp>;

  static inline string GetInputIdStr(int id) { return !id ? "" : std::to_string(id + 1); }
  template <class TypeParam, class TypeParamFirst = TypeParam>
  static void AutoGetIncParams(const string &str_settings_file, vector<TypeParam> &params,
                               const TypeParam &default_param, const bool ballow_empty,
                               const vector<string> &keys = {"Camera", "mode"}) {
    cv::FileStorage fs_settings(str_settings_file.c_str(), cv::FileStorage::READ);
    if (!fs_settings.isOpened()) {
      PRINT_ERR_MUTEX(__FUNCTION__ << " failed to open settings file at: " << str_settings_file << std::endl);
      exit(-1);
    }
    AutoGetIncParams<TypeParam, TypeParamFirst>(fs_settings, params, default_param, ballow_empty, keys);
  }
  template <class TypeParam, class TypeParamFirst = TypeParam>
  static void AutoGetIncParams(const cv::FileStorage &fs_settings, vector<TypeParam> &params,
                               const TypeParam &default_param, const bool ballow_empty,
                               const vector<string> &keys = {"Camera", "mode"});
};

template <class TypeParam, class TypeParamFirst>
void ParamsStorage::AutoGetIncParams(const cv::FileStorage &fs_settings, vector<TypeParam> &params,
                                     const TypeParam &default_param, const bool ballow_empty,
                                     const vector<string> &keys) {
  int i = 0;
  while (1) {
    auto node_tmp = fs_settings[keys[0] + GetInputIdStr(i) + "." + keys[1]];
    if (!node_tmp.empty()) {
      params.push_back((TypeParam)(TypeParamFirst)node_tmp);
    } else
      break;
    ++i;
  }
  if (!ballow_empty && params.empty()) {
    params.push_back(default_param);
  }
}

}  // namespace VIEO_SLAM
