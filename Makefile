.PHONY: test_gemm
test_gemm: test_gemm.cpp
	g++-4.8 -O3 -std=c++11 -fopenmp -mavx2 -mavx -funroll-loops $< -o $@
	./$@
