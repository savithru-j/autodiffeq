INCLUDE( ${CMAKE_SOURCE_DIR}/cmake/ForceOutOfSource.cmake ) #This must be the first thing included

# ADD_SUBDIRECTORY( linearalgebra )
# ADD_SUBDIRECTORY( numerics )

# Create the autodiffeq interface library
add_library(${LIB_NAME} INTERFACE)

# Set compile features
target_compile_features(${LIB_NAME} INTERFACE cxx_std_14)

#Add alias autodiffeq::autodiffeq
add_library(${LIB_NAME}::${LIB_NAME} ALIAS ${LIB_NAME})

# Add the include paths to the target
MESSAGE("PROJECT_SOURCE_DIR: " ${PROJECT_SOURCE_DIR})
# MESSAGE("CMAKE_INSTALL_PREFIX: " ${CMAKE_INSTALL_PREFIX})
# MESSAGE("CMAKE_INSTALL_INCLUDEDIR: " ${CMAKE_INSTALL_INCLUDEDIR})
target_include_directories(${LIB_NAME}
    INTERFACE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Install interface library
install(TARGETS ${LIB_NAME} EXPORT autodiffeqTargets)

# Install header files
install(DIRECTORY ${PROJECT_SOURCE_DIR}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${LIB_NAME} COMPONENT headers
        PATTERN "CMakeLists.txt" EXCLUDE)