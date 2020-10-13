#include <atomic>
#include <iostream>
#include <vector>
#include <thread>
#include <boost/timer/timer.hpp>

// GPU recursive version
__host__ __device__ uint64_t solve(const int N, const int depth = 0, const uint32_t left = 0, const uint32_t mid = 0, const uint32_t right = 0) {
  if (depth == N) return 1;
  uint64_t sum = 0;
  for (uint32_t pos = (((uint32_t)1 << N) - 1) & ~(left | mid | right); pos; pos &= pos-1) {
    uint32_t bit = pos & -pos;
    sum += solve(N, depth+1, (left | bit) << 1, mid | bit, (right | bit) >> 1);
  }
  return sum;
}

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

// GPU non-recursive version
constexpr int threadsPerBlockNonRec = 128;
constexpr int blocksPerGridNonRec = 4096;

__global__ void kernel_ver2(const int N, const int depth,
    const uint32_t * const left_ary,
    const uint32_t * const mid_ary,
    const uint32_t * const right_ary,
    uint64_t * const result_ary,
    const size_t size) {
  constexpr size_t stackSize = 16;
  __shared__ uint32_t stack_modified[stackSize][threadsPerBlockNonRec];
  __shared__ uint32_t stack_pos[stackSize][threadsPerBlockNonRec];
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
  kernel<<<blockCount, threadsPerBlock>>>(N, M, left_ary_d, mid_ary_d, right_ary_d, result_d, length);
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
  std::cerr << "Expanded: " << length << std::endl;
  uint32_t *left_ary_d;
  uint32_t *mid_ary_d;
  uint32_t *right_ary_d;
  cudaMalloc((void**)&left_ary_d, sizeof(uint32_t) * length);
  cudaMalloc((void**)&mid_ary_d, sizeof(uint32_t) * length);
  cudaMalloc((void**)&right_ary_d, sizeof(uint32_t) * length);
  uint64_t *result_d;
  size_t threadsPerGrid = blocksPerGridNonRec * threadsPerBlockNonRec;
  cudaMalloc((void**)&result_d, sizeof(uint64_t) * threadsPerGrid);
  cudaMemcpy(left_ary_d, nqe.left_ary.data(), sizeof(uint32_t) * length, cudaMemcpyHostToDevice);
  cudaMemcpy(mid_ary_d, nqe.mid_ary.data(), sizeof(uint32_t) * length, cudaMemcpyHostToDevice);
  cudaMemcpy(right_ary_d, nqe.right_ary.data(), sizeof(uint32_t) * length, cudaMemcpyHostToDevice);
  cudaFuncSetAttribute(kernel_ver2, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  kernel_ver2<<<blocksPerGridNonRec, threadsPerBlockNonRec>>>(N, M, left_ary_d, mid_ary_d, right_ary_d, result_d, length);
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
  if (argc < 3) {
    std::cerr << "usage: " << argv[0] << " N M" << std::endl;
    return 1;
  }
  const int N = atoi(argv[1]);
  const int M = atoi(argv[2]);
  boost::timer::cpu_timer timer;
  //std::cout << "CPU(naive): " << solve(N) << std::endl;
  //std::cout << timer.format() << std::endl;
  //timer.start();
  //std::cout << "CPU(parallel): " << solve_parallel(N, M) << std::endl;
  //std::cout << timer.format() << std::endl;
  //timer.start();
  //std::cout << "GPU(parallel): " << solve_gpu(N, M) << std::endl;
  //std::cout << timer.format() << std::endl;
  timer.start();
  std::cout << "GPU(optimized): " << solve_gpu_ver2(N, M) << std::endl;
  std::cout << timer.format() << std::endl;
  return 0;
}

