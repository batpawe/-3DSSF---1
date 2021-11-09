cmake_minimum_required(VERSION 3.20)
project(3DSSF_ASSIGNMENT_1)

set(CMAKE_CXX_STANDARD 23)
find_package( OpenCV REQUIRED )
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
set(SOURCE_FILES
        src/main.cpp)
INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR}/include)
add_executable(${PROJECT_NAME} ${SOURCE_FILES} )
include_directories(${OpenCV_INCLUDE_DIRS})
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS})

