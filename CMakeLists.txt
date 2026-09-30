cmake_minimum_required(VERSION 3.16)
project(timestamped_transcriber LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(whisper CONFIG REQUIRED)

add_executable(transcribe_cpp transcribe.cpp)
target_link_libraries(transcribe_cpp PRIVATE whisper)
