INCLUDE( cmake/ForceOutOfSource.cmake ) #This must be the first thing included

#Get the name of the binary directory
STRING( TOUPPER ${CMAKE_BINARY_DIR} BIN_DIR_NAME )

CMAKE_MINIMUM_REQUIRED(VERSION 3.5)
PROJECT(autodiffeq VERSION 0.5.0 LANGUAGES CXX)

SET(CMAKE_CXX_STANDARD 14) 
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_CXX_EXTENSIONS OFF) #without compiler extensions like gnu++11

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  set(IS_TOPLEVEL_PROJECT TRUE)
else()
  set(IS_TOPLEVEL_PROJECT FALSE)
endif()

option(AUTODIFFEQ_INSTALL_LIBRARY "Install library into default locations" ${IS_TOPLEVEL_PROJECT})
option(AUTODIFFEQ_BUILD_TESTS "Build and run tests" ${IS_TOPLEVEL_PROJECT})
option(AUTODIFFEQ_BUILD_EXAMPLES "Build examples" ${IS_TOPLEVEL_PROJECT})

OPTION(AUTODIFFEQ_ENABLE_OPENMP "Enable OpenMP" ON )
IF(AUTODIFFEQ_ENABLE_OPENMP)
  find_package(OpenMP)
  if (OpenMP_CXX_FOUND)
    add_definitions(-DENABLE_OPENMP)
  else()
    MESSAGE("OpenMP not found. Disabled option.")
    set(AUTODIFFEQ_ENABLE_OPENMP OFF CACHE BOOL "Enable OpenMP" FORCE)
  endif()
ENDIF()

OPTION(AUTODIFFEQ_ENABLE_CUDA "Enable CUDA" ON )
if(AUTODIFFEQ_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    add_definitions(-DENABLE_CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_ARCHITECTURES "70;86")
  else()
    MESSAGE("CUDA not found. Disabled option.")
    set(AUTODIFFEQ_ENABLE_CUDA OFF CACHE BOOL "Enable CUDA" FORCE)
  endif()
endif()

INCLUDE(CheckCXXSourceCompiles)
INCLUDE(CheckCXXSourceRuns)

# Include the cmake variables with values for installation directories
INCLUDE(GNUInstallDirs)

INCLUDE(cmake/CompileFlags.cmake)
   
ADD_SUBDIRECTORY(autodiffeq)

if(AUTODIFFEQ_BUILD_EXAMPLES)
  ADD_SUBDIRECTORY(examples)
endif()

if(AUTODIFFEQ_BUILD_TESTS)
  #Enable testing for the project
  enable_testing()

  # Configure google-test an external library.
  ADD_SUBDIRECTORY(${CMAKE_SOURCE_DIR}/externals)
  INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/externals)

  INCLUDE_DIRECTORIES(${GTEST_INCLUDE_DIR})
  ADD_SUBDIRECTORY(test)
endif()

if(AUTODIFFEQ_INSTALL_LIBRARY)
  # Install the cmake config files that permit users to use find_package(autodiff)
  INCLUDE(cmake/autodiffeqInstallCMakeConfigFiles.cmake)
endif()


