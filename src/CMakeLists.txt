# Generic test that uses conan libs

add_subdirectory(python)

add_executable(intro main.cpp)
target_link_libraries(
  intro
  PRIVATE project_options
          CONAN_PKG::fmt
          CONAN_PKG::glog
	  CONAN_PKG::ceres-solver)
	  
