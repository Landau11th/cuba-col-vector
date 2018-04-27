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

	Deng::CUDA_Vec::Col a(10);
	Deng::CUDA_Vec::Col b(10);

	auto c = -b;

	return EXIT_SUCCESS;
}