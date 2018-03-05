solve: solve_gpu.cu
	nvcc -o $@ $^ -O3 -lboost_timer -lboost_system
