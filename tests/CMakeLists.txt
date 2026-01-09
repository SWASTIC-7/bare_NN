function(add_cuda_test)
    cmake_parse_arguments(TEST "" "NAME" "SOURCES;LIBS" ${ARGN})
    
    add_executable(${TEST_NAME} ${TEST_SOURCES})
    
    target_include_directories(${TEST_NAME} PRIVATE
        ${INCLUDE_DIR}
        ${PROJECT_ROOT}
    )
    
    target_link_libraries(${TEST_NAME} PRIVATE
        CUDA::cuda_driver
        CUDA::cudart
        ${TEST_LIBS}
    )
    
    set_target_properties(${TEST_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests
        VS_DEBUGGER_WORKING_DIRECTORY ${PROJECT_ROOT}
    )
    
    add_custom_command(TARGET ${TEST_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${PTX_DIR}
            ${CMAKE_BINARY_DIR}/tests/ptx
        COMMENT "Copying PTX files for ${TEST_NAME}"
    )
    
    add_test(
        NAME ${TEST_NAME}
        COMMAND ${TEST_NAME}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests
    )
endfunction()

add_cuda_test(
    NAME test_vector_op
    SOURCES vector_op.cu
)

add_cuda_test(
    NAME test_matmul
    SOURCES matmul.cu
)

add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -C $<CONFIG>
    DEPENDS test_vector_op test_matmul
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running all tests..."
)
