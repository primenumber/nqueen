NVCC_FLAGS:=-O2 -arch=sm_120

solve: solve_gpu.cu
	nvcc -o $@ $^ $(NVCC_FLAGS) -Xcompiler -fopenmp

solve_ss: solve_gpu_small_shared.cu
	nvcc -o $@ $^ $(NVCC_FLAGS)

.PHONY: clean
clean:
	-rm *.o solve
