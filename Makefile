.PHONY: test_gemm
test_gemm: test_gemm.cpp
	g++-4.8 -g -std=c++11 -fopenmp -funroll-loops -mavx2 -mavx $< -o $@
	./$@
