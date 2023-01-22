# The path where cmake config files are installed
set(LIB_INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/${LIB_NAME})

install(EXPORT autodiffeqTargets
    FILE autodiffeqTargets.cmake
    NAMESPACE ${LIB_NAME}::
    DESTINATION ${LIB_INSTALL_CONFIGDIR}
    COMPONENT cmake)

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/autodiffeqConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
    ARCH_INDEPENDENT)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/autodiffeqConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/autodiffeqConfig.cmake
    INSTALL_DESTINATION ${LIB_INSTALL_CONFIGDIR}
    PATH_VARS LIB_INSTALL_CONFIGDIR)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/autodiffeqConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/autodiffeqConfigVersion.cmake
    DESTINATION ${LIB_INSTALL_CONFIGDIR})
