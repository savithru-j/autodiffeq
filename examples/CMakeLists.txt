INCLUDE( ${CMAKE_SOURCE_DIR}/cmake/ForceOutOfSource.cmake ) #This must be the first thing included

#Get all test files
file(GLOB_RECURSE EXAMPLE_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

MESSAGE("Example files:" ${EXAMPLE_SRCS})

#add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND})

#Run through each source
foreach(exampleSrc ${EXAMPLE_SRCS})
        #Extract the filename without an extension (NAME_WE)
        get_filename_component(exampleName ${exampleSrc} NAME_WE)

        #Add compile target
        add_executable(${exampleName} ${exampleSrc})

        #link to libraries and dependencies
        target_link_libraries(${exampleName} PRIVATE ${PROJECT_NAME}::${PROJECT_NAME})

endforeach()


