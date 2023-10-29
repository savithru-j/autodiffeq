INCLUDE( ${CMAKE_SOURCE_DIR}/cmake/ForceOutOfSource.cmake ) #This must be the first thing included

# Create the autodiffeq interface library
add_library(autodiffeq INTERFACE)

# Set compile features
target_compile_features(autodiffeq INTERFACE cxx_std_14)

#Add alias autodiffeq::autodiffeq
add_library(autodiffeq::autodiffeq ALIAS autodiffeq)

# Add the include paths to the target
target_include_directories(
    autodiffeq INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
                         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)