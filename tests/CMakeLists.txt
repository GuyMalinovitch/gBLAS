set(TARGET gBLAS_tests)
include(FetchContent)
include(GoogleTest)

FetchContent_Declare(googletest
                     URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

FetchContent_MakeAvailable(googletest)
enable_testing()


file(GLOB_RECURSE test_files ${CMAKE_SOURCE_DIR}/tests/*.cpp)

add_executable(${TARGET} ${test_files})
target_link_libraries(${TARGET} gBLAS GTest::gtest_main)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}
                                             ${CMAKE_SOURCE_DIR}/src )
target_compile_options(${TARGET} PRIVATE -fpermissive)

gtest_discover_tests(${TARGET})