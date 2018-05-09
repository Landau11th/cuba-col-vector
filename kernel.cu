//C or C++ headers
#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include<iostream>
#include<cmath>

//CUDA headers
#include <cuda.h>
#include <curand.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
////CUDA thrust
//#include <thrust/host_vector.h>
//#include <thrust/device_vector.h>
//#include <thrust/copy.h>
//#include <thrust/for_each.h>

#include"cublas_wrapper.h"

int main(int argc, char * argv[])
{
	const size_t n = 32;
	float* vec_host = new float[n];
	std::cout << vec_host << std::endl;
	for (size_t i = 0; i < n; ++i)
	{
		vec_host[i] = i;
	}
	
	Deng::CUDA_Vec::Col a(vec_host, n);
	Deng::CUDA_Vec::Col b(vec_host, n);
	b = (-1.5)*a;

	std::cout << b.size() << std::endl;
	float* out = b.to_host();

	std::cout << out << std::endl;
	for (int i = 0; i < n; ++i)
	{
		std::cout << out[i] << ", ";
	}

	return EXIT_SUCCESS;
}