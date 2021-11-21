#include <atomic>
#include <iostream>
#include <vector>
#include <thread>
#include <boost/timer/timer.hpp>

// common
__host__ __device__ uint64_t solve(const int N, const int depth = 0, const uint32_t left = 0, const uint32_t mid = 0, const uint32_t right = 0) {
  if (depth == N) return 1;
  uint64_t sum = 0;
  for (uint32_t pos = (((uint32_t)1 << N) - 1) & ~(left | mid | right); pos; pos &= pos-1) {
    uint32_t bit = pos & -pos;
    sum += solve(N, depth+1, (left | bit) << 1, mid | bit, (right | bit) >> 1);
  }
  return sum;
}

// GPU recursive version
namespace gpu_ver1 {

__global__ void kernel(const int N, const int depth,
    const uint32_t * const left_ary,
    const uint32_t * const mid_ary,
    const uint32_t * const right_ary,
    uint64_t * const result_ary,
    const size_t size) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < size) {
    result_ary[index] = solve(N, depth, left_ary[index], mid_ary[index], right_ary[index]);
  }
}

} // namespace gpu_ver1

// GPU non-recursive version
namespace gpu_ver2 {

constexpr int threadsPerBlock = 128;
constexpr int blocksPerGrid = 4096;

__global__ void kernel(const int N, const int depth,
    const uint32_t * const left_ary,
    const uint32_t * const mid_ary,
    const uint32_t * const right_ary,
    uint64_t * const result_ary,
    const size_t size) {
  constexpr size_t stackSize = 16;
  __shared__ uint32_t stack_modified[stackSize][threadsPerBlock];
  __shared__ uint32_t stack_pos[stackSize][threadsPerBlock];
  size_t offset = threadIdx.x + blockIdx.x * blockDim.x;
  size_t stride = gridDim.x * blockDim.x;
  size_t index = offset;
  if (index >= size) {
    result_ary[offset] = 0;
    return;
  }
  size_t stack_index = 0;
  const uint32_t mask_N = ((uint32_t)1 << N) - 1;
  uint64_t count = 0;
  int right_shift = 64 - N;
  uint64_t left = left_ary[index], right = static_cast<uint64_t>(right_ary[index]) << right_shift;
  uint32_t mid = mid_ary[index];
  stack_modified[0][threadIdx.x] = 0;
  stack_pos[0][threadIdx.x] = mask_N & ~(left | mid | (right >> right_shift));
  while (true) {
    if (depth + stack_index == N) {
      ++count;
    }
    uint32_t pos = stack_pos[stack_index][threadIdx.x];
    if (pos == 0) {
      if (stack_index == 0) {
        index += stride;
        if (index >= size) {
          result_ary[offset] = count;
          return;
        } else {
          left = left_ary[index];
          mid = mid_ary[index];
          right = static_cast<uint64_t>(right_ary[index]) << right_shift;
          //stack_modified[0][threadIdx.x] = 0; /* always zero */
          stack_pos[0][threadIdx.x] = mask_N & ~(left | mid | (right >> right_shift));
        }
      } else {
        uint64_t bit = stack_modified[stack_index][threadIdx.x];
        left = (left >> 1) ^ bit;
        mid ^= bit;
        right = (right << 1) ^ (bit << right_shift);
        --stack_index;
      }
    } else {
      uint64_t bit = pos & -pos;
      stack_pos[stack_index][threadIdx.x] = pos ^ bit;
      left = (left | bit) << 1;
      mid |= bit;
      right = (right | (bit << right_shift)) >> 1;
      ++stack_index;
      stack_modified[stack_index][threadIdx.x] = bit;
      stack_pos[stack_index][threadIdx.x] = mask_N & ~(left | mid | (right >> right_shift));
    }
  }
}

} // namespace gpu_ver2

// GPU non-recursive, reduced shared memory use
namespace gpu_ver3 {
constexpr int threadsPerBlock = 256;
constexpr int blocksPerGrid = 4096;

__global__ void kernel(const int N, const int depth,
    const uint32_t * const left_ary,
    const uint32_t * const mid_ary,
    const uint32_t * const right_ary,
    uint64_t * const result_ary,
    const size_t size) {
  constexpr size_t stackSize = 16;
  __shared__ uint32_t stack_bits[stackSize][threadsPerBlock];
  size_t offset = threadIdx.x + blockIdx.x * blockDim.x;
  size_t stride = gridDim.x * blockDim.x;
  size_t index = offset;
  if (index >= size) {
    result_ary[offset] = 0;
    return;
  }
  size_t stack_index = 0;
  const uint32_t mask_N = ((uint32_t)1 << N) - 1;
  uint64_t count = 0;
  int right_shift = 64 - N;
  uint64_t left = left_ary[index], mid = mid_ary[index], right = static_cast<uint64_t>(right_ary[index]) << right_shift;
  stack_bits[0][threadIdx.x] = mask_N & ~(left | mid | (right >> right_shift));
  while (true) {
    if (depth + stack_index == N) {
      ++count;
    }
    uint32_t bits = stack_bits[stack_index][threadIdx.x];
    uint32_t mask = mask_N & ~(left | mid | (right >> right_shift));
    uint32_t pos = bits & mask;
    uint64_t modified = bits & ~mask;
    if (pos == 0) {
      if (stack_index == 0) {
        index += stride;
        if (index >= size) {
          result_ary[offset] = count;
          return;
        } else {
          left = left_ary[index];
          mid = mid_ary[index];
          right = static_cast<uint64_t>(right_ary[index]) << right_shift;
          stack_bits[0][threadIdx.x] = mask_N & ~(left | mid | (right >> right_shift));
        }
      } else {
        left = (left >> 1) ^ modified;
        mid ^= modified;
        right = (right << 1) ^ (modified << right_shift);
        --stack_index;
      }
    } else {
      uint64_t bit = pos & -pos;
      stack_bits[stack_index][threadIdx.x] = bits ^ bit;
      left = (left | bit) << 1;
      mid |= bit;
      right = (right | (bit << right_shift)) >> 1;
      ++stack_index;
      stack_bits[stack_index][threadIdx.x] = bit | (mask_N & ~(left | mid | (right >> right_shift)));
    }
  }
}

} // namespace gpu_ver3

class NQueenExpand {
 public:
  std::vector<uint32_t> left_ary, mid_ary, right_ary;
  void expand(int N, int M, int depth = 0, uint32_t left = 0, uint32_t mid = 0, uint32_t right = 0) {
    if (depth == M) {
      left_ary.push_back(left);
      mid_ary.push_back(mid);
      right_ary.push_back(right);
      return;
    }
    for (uint32_t pos = (((uint32_t)1 << N) - 1) & ~(left | mid | right); pos; pos &= pos-1) {
      uint32_t bit = pos & -pos;
      expand(N, M, depth+1, (left | bit) << 1, mid | bit, (right | bit) >> 1);
    }
  }
};

uint64_t solve_parallel(const int N, const int M) {
  NQueenExpand nqe;
  nqe.expand(N, M);
  const size_t length = nqe.left_ary.size();
  std::atomic<size_t> index(0);
  std::atomic<uint64_t> sum(0);
  std::vector<std::thread> vt;
  for (size_t i = 0; i < std::thread::hardware_concurrency(); ++i) {
    vt.emplace_back([&] {
      while (true) {
        size_t local_index = index++;
        if (local_index >= length) return;
        sum += solve(N, M, nqe.left_ary[local_index], nqe.mid_ary[local_index], nqe.right_ary[local_index]);
      }
    });
  }
  for (auto &&t : vt) t.join();
  return sum;
}

uint64_t solve_gpu(const int N, const int M) {
  NQueenExpand nqe;
  nqe.expand(N, M);
  const size_t length = nqe.left_ary.size();
  uint32_t *left_ary_d;
  uint32_t *mid_ary_d;
  uint32_t *right_ary_d;
  cudaMalloc((void**)&left_ary_d, sizeof(uint32_t) * length);
  cudaMalloc((void**)&mid_ary_d, sizeof(uint32_t) * length);
  cudaMalloc((void**)&right_ary_d, sizeof(uint32_t) * length);
  uint64_t *result_d;
  cudaMalloc((void**)&result_d, sizeof(uint64_t) * length);
  cudaMemcpy(left_ary_d, nqe.left_ary.data(), sizeof(uint32_t) * length, cudaMemcpyHostToDevice);
  cudaMemcpy(mid_ary_d, nqe.mid_ary.data(), sizeof(uint32_t) * length, cudaMemcpyHostToDevice);
  cudaMemcpy(right_ary_d, nqe.right_ary.data(), sizeof(uint32_t) * length, cudaMemcpyHostToDevice);
  constexpr int threadsPerBlock = 128;
  const int blockCount = (length + threadsPerBlock - 1) / threadsPerBlock;
  gpu_ver1::kernel<<<blockCount, threadsPerBlock>>>(N, M, left_ary_d, mid_ary_d, right_ary_d, result_d, length);
  std::vector<uint64_t> result(length);
  cudaMemcpy(result.data(), result_d, sizeof(uint64_t) * length, cudaMemcpyDeviceToHost);
  uint64_t sum = 0;
  for (size_t i = 0; i < length; ++i) sum += result[i];
  cudaFree(left_ary_d);
  cudaFree(mid_ary_d);
  cudaFree(right_ary_d);
  cudaFree(result_d);
  return sum;
}

uint64_t solve_gpu_ver2(const int N, const int M) {
  NQueenExpand nqe;
  nqe.expand(N, M);
  const size_t length = nqe.left_ary.size();
  uint32_t *left_ary_d;
  uint32_t *mid_ary_d;
  uint32_t *right_ary_d;
  cudaMalloc((void**)&left_ary_d, sizeof(uint32_t) * length);
  cudaMalloc((void**)&mid_ary_d, sizeof(uint32_t) * length);
  cudaMalloc((void**)&right_ary_d, sizeof(uint32_t) * length);
  uint64_t *result_d;
  size_t threadsPerGrid = gpu_ver2::blocksPerGrid * gpu_ver2::threadsPerBlock;
  cudaMalloc((void**)&result_d, sizeof(uint64_t) * threadsPerGrid);
  cudaMemcpy(left_ary_d, nqe.left_ary.data(), sizeof(uint32_t) * length, cudaMemcpyHostToDevice);
  cudaMemcpy(mid_ary_d, nqe.mid_ary.data(), sizeof(uint32_t) * length, cudaMemcpyHostToDevice);
  cudaMemcpy(right_ary_d, nqe.right_ary.data(), sizeof(uint32_t) * length, cudaMemcpyHostToDevice);
  cudaFuncSetAttribute(gpu_ver2::kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 90);
  gpu_ver2::kernel<<<gpu_ver2::blocksPerGrid, gpu_ver2::threadsPerBlock>>>(N, M, left_ary_d, mid_ary_d, right_ary_d, result_d, length);
  std::vector<uint64_t> result(threadsPerGrid);
  cudaMemcpy(result.data(), result_d, sizeof(uint64_t) * threadsPerGrid, cudaMemcpyDeviceToHost);
  uint64_t sum = 0;
  for (size_t i = 0; i < threadsPerGrid; ++i) sum += result[i];
  cudaFree(left_ary_d);
  cudaFree(mid_ary_d);
  cudaFree(right_ary_d);
  cudaFree(result_d);
  return sum;
}

uint64_t solve_gpu_ver3(const int N, const int M) {
  NQueenExpand nqe;
  nqe.expand(N, M);
  const size_t length = nqe.left_ary.size();
  uint32_t *left_ary_d;
  uint32_t *mid_ary_d;
  uint32_t *right_ary_d;
  cudaMalloc((void**)&left_ary_d, sizeof(uint32_t) * length);
  cudaMalloc((void**)&mid_ary_d, sizeof(uint32_t) * length);
  cudaMalloc((void**)&right_ary_d, sizeof(uint32_t) * length);
  uint64_t *result_d;
  size_t threadsPerGrid = gpu_ver3::blocksPerGrid * gpu_ver3::threadsPerBlock;
  cudaMalloc((void**)&result_d, sizeof(uint64_t) * threadsPerGrid);
  cudaMemcpy(left_ary_d, nqe.left_ary.data(), sizeof(uint32_t) * length, cudaMemcpyHostToDevice);
  cudaMemcpy(mid_ary_d, nqe.mid_ary.data(), sizeof(uint32_t) * length, cudaMemcpyHostToDevice);
  cudaMemcpy(right_ary_d, nqe.right_ary.data(), sizeof(uint32_t) * length, cudaMemcpyHostToDevice);
  cudaFuncSetAttribute(gpu_ver3::kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 75);
  gpu_ver3::kernel<<<gpu_ver3::blocksPerGrid, gpu_ver3::threadsPerBlock>>>(N, M, left_ary_d, mid_ary_d, right_ary_d, result_d, length);
  std::vector<uint64_t> result(threadsPerGrid);
  cudaMemcpy(result.data(), result_d, sizeof(uint64_t) * threadsPerGrid, cudaMemcpyDeviceToHost);
  uint64_t sum = 0;
  for (size_t i = 0; i < threadsPerGrid; ++i) sum += result[i];
  cudaFree(left_ary_d);
  cudaFree(mid_ary_d);
  cudaFree(right_ary_d);
  cudaFree(result_d);
  return sum;
}

int main(int argc, char **argv) {
  if (argc < 4) {
    std::cerr << "usage: " << argv[0] << " SOLVER N M" << std::endl;
    std::cerr << "SOLVER: CPU_NAIVE | CPU_PARALLEL | GPU_PARALLEL | GPU_OPTIMIZED" << std::endl;
    return 1;
  }
  std::string solver = argv[1];
  const int N = atoi(argv[2]);
  const int M = atoi(argv[3]);
  boost::timer::cpu_timer timer;
  if (solver == "CPU_NAIVE") {
    std::cout << "CPU(naive): " << solve(N) << std::endl;
    std::cout << timer.format() << std::endl;
  } else if (solver == "CPU_PARALLEL") {
    std::cout << "CPU(parallel): " << solve_parallel(N, M) << std::endl;
    std::cout << timer.format() << std::endl;
  } else if (solver == "GPU_PARALLEL") {
    std::cout << "GPU(parallel): " << solve_gpu(N, M) << std::endl;
    std::cout << timer.format() << std::endl;
  } else if (solver == "GPU_OPTIMIZED") {
    std::cout << "GPU(optimized): " << solve_gpu_ver2(N, M) << std::endl;
    std::cout << timer.format() << std::endl;
  } else if (solver == "GPU_TESTING") {
    std::cout << "GPU(testing): " << solve_gpu_ver3(N, M) << std::endl;
    std::cout << timer.format() << std::endl;
  } else {
    std::cerr << "Unknown solver: " << solver << std::endl;
    return EXIT_FAILURE;
  }
  return 0;
}

