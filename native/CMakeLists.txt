cmake_minimum_required(VERSION 3.18)
project(pipnn_native LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_package(Threads REQUIRED)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(ABSL_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_CONTRIB ON CACHE BOOL "" FORCE)
add_subdirectory(third_party/abseil EXCLUDE_FROM_ALL)
add_subdirectory(third_party/highway EXCLUDE_FROM_ALL)
add_executable(ann_bench src/bench.cpp)
target_include_directories(ann_bench PRIVATE
  third_party/PiPNN/algorithms third_party/PiPNN/parlaylib/include
  third_party/eigen third_party/hnswlib)
target_compile_definitions(ann_bench PRIVATE EIGEN_DONT_PARALLELIZE)
target_compile_options(ann_bench PRIVATE -O3 -march=native -ffast-math)
target_link_libraries(ann_bench PRIVATE Threads::Threads
  absl::synchronization absl::flat_hash_set absl::span hwy hwy_contrib)

add_executable(mutation_test src/mutation_test.cpp)
target_include_directories(mutation_test PRIVATE third_party/PiPNN/algorithms third_party/PiPNN/parlaylib/include third_party/eigen)
target_compile_definitions(mutation_test PRIVATE EIGEN_DONT_PARALLELIZE)
target_compile_options(mutation_test PRIVATE -O3 -march=native -ffast-math)
target_link_libraries(mutation_test PRIVATE Threads::Threads absl::synchronization absl::flat_hash_set absl::span hwy hwy_contrib)
# From-scratch PiPNN: needs only Eigen and threads.
add_executable(fastpipnn_test src/fastpipnn_test.cpp)
target_include_directories(fastpipnn_test PRIVATE third_party/eigen)
target_compile_definitions(fastpipnn_test PRIVATE EIGEN_DONT_PARALLELIZE)
target_compile_options(fastpipnn_test PRIVATE -O3 -march=native -ffast-math)
target_link_libraries(fastpipnn_test PRIVATE Threads::Threads)
# Graph export for the Python filtered-search experiments.
add_executable(pipnn_graph src/pipnn_graph.cpp)
target_include_directories(pipnn_graph PRIVATE third_party/eigen)
target_compile_definitions(pipnn_graph PRIVATE EIGEN_DONT_PARALLELIZE)
target_compile_options(pipnn_graph PRIVATE -O3 -march=native -ffast-math)
target_link_libraries(pipnn_graph PRIVATE Threads::Threads)

enable_testing()
add_test(NAME native_mutations COMMAND mutation_test)
add_test(NAME fastpipnn COMMAND fastpipnn_test)

option(NATIVE_SANITIZERS "Instrument native targets for correctness checks" OFF)
if(NATIVE_SANITIZERS)
  foreach(target ann_bench mutation_test fastpipnn_test)
    target_compile_options(${target} PRIVATE -O1 -fno-fast-math -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(${target} PRIVATE -fsanitize=address,undefined)
  endforeach()
endif()

# Sharded designs (D1 graph shards, D5/D6 IVF segments, router study).
add_executable(shard_bench src/shard_bench.cpp)
target_include_directories(shard_bench PRIVATE third_party/eigen third_party/hnswlib)
target_compile_definitions(shard_bench PRIVATE EIGEN_DONT_PARALLELIZE)
target_compile_options(shard_bench PRIVATE -O3 -march=native -ffast-math)
target_link_libraries(shard_bench PRIVATE Threads::Threads)

# On-disk graph index (PiPNN or HNSW level 0 in a DiskANN-style layout).
add_executable(disk_bench src/disk_bench.cpp)
target_include_directories(disk_bench PRIVATE third_party/eigen third_party/hnswlib)
target_compile_definitions(disk_bench PRIVATE EIGEN_DONT_PARALLELIZE)
target_compile_options(disk_bench PRIVATE -O3 -march=native -ffast-math)
target_link_libraries(disk_bench PRIVATE Threads::Threads)

# IVF with graphs for assignment and inside partitions: LanceDB's HNSW vs fastpipnn (IVF_GRAPH.md).
add_executable(ivf_graph_bench src/ivf_graph_bench.cpp)
target_include_directories(ivf_graph_bench PRIVATE third_party/eigen third_party/hnswlib)
target_compile_definitions(ivf_graph_bench PRIVATE EIGEN_DONT_PARALLELIZE)
target_compile_options(ivf_graph_bench PRIVATE -O3 -march=native -ffast-math)
target_link_libraries(ivf_graph_bench PRIVATE Threads::Threads)

# Row -> partition assignment against fixed centroids, for Lance precomputed partitions (IVF_RQ.md).
add_executable(ivf_assign src/ivf_assign.cpp)
target_include_directories(ivf_assign PRIVATE third_party/eigen third_party/hnswlib)
target_compile_definitions(ivf_assign PRIVATE EIGEN_DONT_PARALLELIZE)
target_compile_options(ivf_assign PRIVATE -O3 -march=native -ffast-math)
target_link_libraries(ivf_assign PRIVATE Threads::Threads)
