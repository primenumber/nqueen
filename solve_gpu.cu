#include <atomic>
#include <iostream>
#include <vector>
#include <thread>
#include <boost/timer/timer.hpp>

constexpr int threadsPerBlock = 64;

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

struct Node {
  int depth;
  uint32_t left, mid, right;
  uint32_t pos;
};

__device__ uint64_t solve_nonrec(const int N, const int depth, const uint32_t left, const uint32_t mid, const uint32_t right) {
  int stack_index = 0;
  uint64_t count = 0;
  __shared__ Node stack[threadsPerBlock][16];
  stack[threadIdx.x][0] = {depth, left, mid, right, (((uint32_t)1 << N) - 1) & ~(left | mid | right)};
  while (true) {
    if (stack[threadIdx.x][stack_index].depth == N) {
      ++count;
    }
    uint32_t pos = stack[threadIdx.x][stack_index].pos;
    if (pos == 0) {
      if (stack_index == 0) return count; // end solve
      --stack_index;
    } else {
      uint32_t bit = pos & -pos;
      stack[threadIdx.x][stack_index].pos ^= bit;
      int new_depth = stack[threadIdx.x][stack_index].depth + 1;
      uint32_t new_left = (stack[threadIdx.x][stack_index].left | bit) << 1;
      uint32_t new_mid = stack[threadIdx.x][stack_index].mid | bit;
      uint32_t new_right = (stack[threadIdx.x][stack_index].right | bit) >> 1;
      ++stack_index;
      stack[threadIdx.x][stack_index] = {
        new_depth, new_left, new_mid, new_right,
        (((uint32_t)1 << N) - 1) & ~(new_left | new_mid | new_right)
      };
    }
  }
}

__global__ void kernel_ver2(const int N, const int depth,
    const uint32_t * const left_ary,
    const uint32_t * const mid_ary,
    const uint32_t * const right_ary,
    uint64_t * const result_ary,
    const size_t size) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < size) {
    result_ary[index] = solve_nonrec(N, depth, left_ary[index], mid_ary[index], right_ary[index]);
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
  const int blockCount = (length + threadsPerBlock - 1) / threadsPerBlock;
  kernel_ver2<<<blockCount, threadsPerBlock>>>(N, M, left_ary_d, mid_ary_d, right_ary_d, result_d, length);
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

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "usage: " << argv[0] << " N M" << std::endl;
    return 1;
  }
  const int N = atoi(argv[1]);
  const int M = atoi(argv[2]);
  boost::timer::cpu_timer timer;
  std::cout << "CPU(naive): " << solve(N) << std::endl;
  std::cout << timer.format() << std::endl;
  timer.start();
  std::cout << "CPU(parallel): " << solve_parallel(N, M) << std::endl;
  std::cout << timer.format() << std::endl;
  timer.start();
  std::cout << "GPU(parallel): " << solve_gpu(N, M) << std::endl;
  std::cout << timer.format() << std::endl;
  timer.start();
  std::cout << "GPU(optimized): " << solve_gpu_ver2(N, M) << std::endl;
  std::cout << timer.format() << std::endl;
  return 0;
}

