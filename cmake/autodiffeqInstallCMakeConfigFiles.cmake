  # locations are provided by GNUInstallDirs
  install(
    TARGETS autodiffeq
    EXPORT autodiffeqTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "autodiffeqConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
    ARCH_INDEPENDENT)

configure_package_config_file(
    ${PROJECT_SOURCE_DIR}/cmake/autodiffeqConfig.cmake.in
    ${PROJECT_BINARY_DIR}/autodiffeqConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/autodiffeq/cmake)

install(EXPORT autodiffeqTargets
    FILE autodiffeqTargets.cmake
    NAMESPACE autodiffeq::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/autodiffeq/cmake
    COMPONENT cmake)

install(FILES
    ${PROJECT_BINARY_DIR}/autodiffeqConfig.cmake
    ${PROJECT_BINARY_DIR}/autodiffeqConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/autodiffeq/cmake)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/autodiffeq 
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        PATTERN "CMakeLists.txt" EXCLUDE)