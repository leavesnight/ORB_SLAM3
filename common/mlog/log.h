//
// Created by leavesnight on 2021/12/21.
//

#pragma once

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <mutex>
#include "common/interface.h"
#ifdef NDEBUG
// assert in Release, should be put at the last include <assert.h> in .h/.cpp
#undef NDEBUG
#include <assert.h>
// if still not assert takes effects, annotate this
//#define NDEBUG
#else
#include <assert.h>
#endif

namespace VIEO_SLAM {
namespace mlog {
enum TYPE_PRINT_LEVEL { PRINT_LEVEL_ERROR = 0, PRINT_LEVEL_INFO = 1, PRINT_LEVEL_DEBUG = 2 };
constexpr int PRINT_LEVEL = PRINT_LEVEL_INFO;  // PRINT_LEVEL_DEBUG;  //

typedef enum kVerboseLevel { kVerbRel, kVerbDeb, kVerbFull } ekVerboseLevel;

// to files log related
const std::string vieo_slam_debug_path = "/home/leavesnight/tmp/VIEOSLAM/";
const std::string online_calibrate_debug_path = "/home/leavesnight/tmp/VIEOSLAM/OnlineCalib/";
const std::string hard_case_debug_path = "/home/leavesnight/tmp/VIEOSLAM/HardCase/";

// multi-threads log related
extern COMMON_API std::mutex gmutexOUTPUT;

// macro is invalid for namespace, but put here for Timer definition
#define PRINT_INFO_BASE(msg, level, foldername, filename)              \
  do {                                                                 \
    if (VIEO_SLAM::mlog::PRINT_LEVEL >= level) {                       \
      auto foldername_str = std::string(foldername);                   \
      if (!foldername_str.empty()) {                                   \
        std::string debug_file = foldername_str + filename;            \
        std::ofstream fout(debug_file, std::ios::out | std::ios::app); \
        fout << msg;                                                   \
      } else {                                                         \
        if (VIEO_SLAM::mlog::PRINT_LEVEL_ERROR == level)               \
          std::cerr << msg;                                            \
        else                                                           \
          std::cout << msg;                                            \
      }                                                                \
    }                                                                  \
  } while (0)
#define PRINT_INFO_MUTEX_BASE(msg, level, foldername, filename)           \
  do {                                                                    \
    if (VIEO_SLAM::mlog::PRINT_LEVEL >= level) {                          \
      auto foldername_str = std::string(foldername);                      \
      if (!foldername_str.empty()) {                                      \
        std::unique_lock<std::mutex> lock(VIEO_SLAM::mlog::gmutexOUTPUT); \
        std::string debug_file = foldername_str + filename;               \
        std::ofstream fout(debug_file, std::ios::out | std::ios::app);    \
        fout << msg;                                                      \
      } else {                                                            \
        std::ios oldstate(nullptr);                                       \
        if (VIEO_SLAM::mlog::PRINT_LEVEL_ERROR == level) {                \
          oldstate.copyfmt(std::cerr);                                    \
          std::cerr << msg;                                               \
          std::cerr.copyfmt(oldstate);                                    \
        } else {                                                          \
          oldstate.copyfmt(std::cout);                                    \
          std::cout << msg;                                               \
          std::cout.copyfmt(oldstate);                                    \
        }                                                                 \
      }                                                                   \
    }                                                                     \
  } while (0)
#define PRINT_ERR(msg) PRINT_INFO_BASE(msg, VIEO_SLAM::mlog::PRINT_LEVEL_ERROR, "", "")
#define PRINT_ERR_MUTEX(msg) PRINT_INFO_MUTEX_BASE(msg, VIEO_SLAM::mlog::PRINT_LEVEL_ERROR, "", "")
#define PRINT_INFO(msg) PRINT_INFO_BASE(msg, VIEO_SLAM::mlog::PRINT_LEVEL_INFO, "", "")
#define PRINT_INFO_MUTEX(msg) PRINT_INFO_MUTEX_BASE(msg, VIEO_SLAM::mlog::PRINT_LEVEL_INFO, "", "")
#define PRINT_DEBUG(msg) PRINT_INFO_BASE(msg, VIEO_SLAM::mlog::PRINT_LEVEL_DEBUG, "", "")
#define PRINT_DEBUG_MUTEX(msg) PRINT_INFO_MUTEX_BASE(msg, VIEO_SLAM::mlog::PRINT_LEVEL_DEBUG, "", "")
#define PRINT_INFO_FILE(msg, foldername, filename) \
  PRINT_INFO_BASE(msg, VIEO_SLAM::mlog::PRINT_LEVEL_INFO, foldername, filename)
#define PRINT_INFO_FILE_MUTEX(msg, foldername, filename) \
  PRINT_INFO_MUTEX_BASE(msg, VIEO_SLAM::mlog::PRINT_LEVEL_INFO, foldername, filename)
#define PRINT_DEBUG_FILE(msg, foldername, filename) \
  PRINT_INFO_BASE(msg, VIEO_SLAM::mlog::PRINT_LEVEL_DEBUG, foldername, filename)
#define PRINT_DEBUG_FILE_MUTEX(msg, foldername, filename) \
  PRINT_INFO_MUTEX_BASE(msg, VIEO_SLAM::mlog::PRINT_LEVEL_DEBUG, foldername, filename)
#define CLEAR_INFO_FILE(msg, foldername, filename)        \
  do {                                                    \
    auto foldername_str = std::string(foldername);        \
    if (!foldername_str.empty()) {                        \
      std::string debug_file = foldername_str + filename; \
      std::ofstream fout(debug_file, std::ios::out);      \
      fout << msg;                                        \
    }                                                     \
  } while (0)

template <class T>
class ArgToStream {
 public:
  static void impl(std::stringstream& stream, T&& arg) { stream << std::forward<T>(arg); }
};
// Following: http://stackoverflow.com/a/22759544
template <class T>
class IsStreamable {
 private:
  template <class TT>
  static auto test(int) -> decltype(std::declval<std::stringstream&>() << std::declval<TT>(), std::true_type());

  template <class>
  static auto test(...) -> std::false_type;

 public:
  static bool const value = decltype(test<T>(0))::value;
};
inline void FormatStream(std::stringstream& stream, char const* text) {
  stream << text;
  return;
}
/**
 * Following: http://en.cppreference.com/w/cpp/language/parameter_pack
 * @stream sstream to output
 * @text format string, now support:
 * '{}'->@arg
 * '{.}'->fixed<<@arg
 * '{.Num}'->fixed<<setprecision(Num)<<@arg
 * @arg the next class T param to substitute the next '{...}' in @text
 * @args the next next params to be @arg when calling this func. again by passing @args to be @arg
 */
template <class T, typename... Args>
void FormatStream(std::stringstream& stream, char const* text, T&& arg, Args&&... args) {
  static_assert(IsStreamable<T>::value, "One of the args has no ostream overload!");
  for (; *text != '\0'; ++text) {
    if (*text == '{') {
      if (*(text + 1) == '{') {
        ++text;
      } else {
        bool bend = *(text + 1) == '}';
        std::string str_tmp;
        if (!bend) {
          str_tmp = text + 1;
          auto pos_end = str_tmp.find('}');
          if (std::string::npos != pos_end) {
            str_tmp = str_tmp.substr(0, pos_end);
            text += str_tmp.size();
            bend = true;
          }
        }
        if (bend) {
          if (!str_tmp.empty()) {
            int i = 0, iend = str_tmp.size();
            char c_tmp = str_tmp[i];
            if (c_tmp == '.') stream << std::fixed;
            std::string str_tmp2;
            while (++i < iend && isdigit(str_tmp[i])) {
              c_tmp = str_tmp[i];
              str_tmp2.push_back(c_tmp);
            }
            if (!str_tmp2.empty()) {
              stream << std::setprecision(std::stoi(str_tmp2));
            }
          }
          ArgToStream<T&&>::impl(stream, std::forward<T>(arg));
          FormatStream(stream, text + 2, std::forward<Args>(args)...);
          return;
        }
      }
    }
    stream << *text;
  }
  stream << "\nFormat-Warning: There are " << sizeof...(Args) + 1 << " args unused.";
  return;
}
template <class... Args>
std::string FormatString(char const* text, Args&&... args) {
  std::stringstream stream;
  FormatStream(stream, text, std::forward<Args>(args)...);
  return stream.str();
}
template <class... Args>
void defaultEnsure(char const* function, char const* file, int line, char const* description, Args&&... args) {
  std::printf("mlog ensure failed in function '%s', file '%s', line %d.\n", function, file, line);
#ifdef __CUDACC__
  std::printf("%s", description);
#else
  std::cout << FormatString(description, std::forward<Args>(args)...) << std::endl;
  std::abort();
#endif
}
#define MLOG_ENSURE(expr, ...) \
  ((expr) ? ((void)0) : VIEO_SLAM::mlog::defaultEnsure(__FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__))

// colorful cout related
#define redSTR "\033[31m"
#define brightredSTR "\033[31;1m"
#define greenSTR "\e[32m"
#define brightgreenSTR "\e[32;1m"
#define blueSTR "\e[34m"
#define brightblueSTR "\e[34;1m"
#define yellowSTR "\e[33;1m"
#define brownSTR "\e[33m"
#define azureSTR "\e[36;1m"
#define whiteSTR "\e[0m"

}  // namespace mlog
}  // namespace VIEO_SLAM
