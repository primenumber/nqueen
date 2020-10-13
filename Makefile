solve: solve_gpu.cu
	nvcc -o $@ $^ -O2 -lboost_timer -lboost_system -arch=sm_86 -Xcompiler -fopenmp

solve_ss: solve_gpu_small_shared.cu
	nvcc -o $@ $^ -O2 -lboost_timer -lboost_system -arch=sm_86

.PHONY: clean
clean:
	-rm *.o solve
