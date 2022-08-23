cmake_minimum_required(VERSION 3.10)
project(tftn)
set(CMAKE_BUILD_TYPE Debug)

string(ASCII 27 Esc)
set(ColourReset "${Esc}[m")
set(ColourBold  "${Esc}[1m")
set(Red         "${Esc}[31m")
set(Green       "${Esc}[32m")
set(Yellow      "${Esc}[33m")
set(Blue        "${Esc}[34m")
set(Magenta     "${Esc}[35m")
set(Cyan        "${Esc}[36m")
set(White       "${Esc}[37m")
set(BoldRed     "${Esc}[1;31m")
set(BoldGreen   "${Esc}[1;32m")
set(BoldYellow  "${Esc}[1;33m")
set(BoldBlue    "${Esc}[1;34m")
set(BoldMagenta "${Esc}[1;35m")
set(BoldCyan    "${Esc}[1;36m")
set(BoldWhite   "${Esc}[1;37m")

#you can change this version
find_package(OpenCV  REQUIRED)
find_package(Boost COMPONENTS system thread program_options)



set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_FLAGS_RELEASE "-march=native -mfma -msse2 -msse3 -msse4")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -march=native -mfma -msse2 -msse3 -msse4")



add_definitions(-std=c++17)
add_definitions(-mfma)
add_definitions(-mno-avx512f)
add_definitions(-mno-avx512vl)
add_definitions(-DUSE_SSE_AUTO)
add_definitions(-DSSE_MATHFUN_WITH_CODE)
set(CPU_ALL_OPTIMIZATIONS "SSE;SSE2;SSE3;SSSE3;SSE4_1;SSE4_2;POPCNT;AVX;FP16;AVX2;FMA3;AVX_512F;AVX512VL;AVX512F;AVX_512VL")
#SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -mavx512f -mfma -mavx512vl -mavx512bw -mavx512dq -march=knl -march=sandybridge -march=haswell")
#message('0-0000000000000000000' ${AVX_FLAGS})

#ADD_DEFINITIONS(-DEIGEN_ENABLE_AVX512=ON) #maybe we dont need eigen, i fortget it.

#create static .a file.
# 一个不错的cmake教程 https://blog.csdn.net/u012564117/article/details/95085360
# 另一个不错的cmake教程 https://zhuanlan.zhihu.com/p/87738686
file(GLOB ALL_HEADERS include/tftn/*.h)
file(GLOB ALL_SOURCES include/*/*.cpp)
#aux_source_directory(include/tftfn/ ALL_SOURCES )

message("${BoldBlue}ALL_HEADERS: ${ColourReset}")
message(${ALL_HEADERS})
message(${BoldBlue}ALL_SOURCES: ${ColourReset})
message(${ALL_SOURCES})



add_library(tftn_static         STATIC   include/tftn/tftn.cpp )
add_library(tftn_dynamic SHARED include/tftn/tftn.cpp  )


SET_TARGET_PROPERTIES(tftn_static PROPERTIES OUTPUT_NAME "tftn") #modify the static file name
SET_TARGET_PROPERTIES(tftn_dynamic PROPERTIES OUTPUT_NAME "tftn") #modify the static file name


include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${PROJECT_NAME}
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/ThirdParty
        ${PYTHON_INCLUDE_DIR}
        ${pybind11_INCLUDE_DIRS}
)

add_executable(3f2n ${CMAKE_CURRENT_SOURCE_DIR}/example/demo.cpp)

target_link_libraries(3f2n tftn_dynamic ${OpenCV_LIBS})