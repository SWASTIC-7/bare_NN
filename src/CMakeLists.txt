add_executable(main main.cu)

target_include_directories(main PRIVATE
    ${INCLUDE_DIR}
)

target_link_libraries(main PRIVATE
    CUDA::cuda_driver
    CUDA::cudart
)

set_target_properties(main PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_ROOT}/build
    VS_DEBUGGER_WORKING_DIRECTORY ${PROJECT_ROOT}
)

add_custom_command(TARGET main POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${PTX_DIR}
        $<TARGET_FILE_DIR:main>/ptx
    COMMENT "Copying PTX files to build directory"
)
