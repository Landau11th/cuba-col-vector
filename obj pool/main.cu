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
	const size_t n = 1024*64;
	const size_t threads_per_block = 512*3;
	
	const size_t N_t = 1024 * 16;
	const float tau = 4.0;
	const float dt = tau / N_t;
	const float dt_sqrt = sqrt(dt);

	float beta = 1.0;
	float minus_beta = -beta;

	float omega_0 = 1.0;
	float omega_tau = 3.0;
	float* omegas = new float[N_t + 1];
	for (size_t i = 0; i <= N_t; ++i)
	{
		omegas[i] = omega_0 + i*(omega_tau - omega_0) / N_t;
	}

	float gamma = 1.0 / 4.0;
	float gamma_dt = -gamma*dt;
	float C = sqrt(2 * gamma / beta);
	
	
	cudaEvent_t start_cuda, stop_cuda;
	cudaEventCreate(&start_cuda);
	cudaEventCreate(&stop_cuda);
	float ellapese_time;

	std::cout << "Start\n";
	cudaEventRecord(start_cuda, 0);

	Deng::CUDA_Vec::rand_seed();

	Deng::CUDA_Vec::Col W(n);
	W.zeros();

	Deng::CUDA_Vec::Col X = Deng::CUDA_Vec::rand_normal(n, 0.0, 1 / (sqrt(beta)*omega_0));
	Deng::CUDA_Vec::Col V = Deng::CUDA_Vec::rand_normal(n, 0.0, 1 / sqrt(beta));

	for (size_t i = 0; i < N_t; ++i)
	{
		W += (-dt*omegas[i] * (omegas[i + 1] - omegas[i]))*(X*X);
		auto dV = (-dt*omegas[i] * omegas[i])*X + (-gamma*dt)*V + C*Deng::CUDA_Vec::rand_normal(n, 0.0, dt_sqrt);
		X += dt*V;
		V += dV;
	}
	

	//std::cout << b.size() << std::endl;
	//float* out = b.to_host();

	//std::cout << out << std::endl;
	//for (int i = 0; i < n; ++i)
	//{
	//	std::cout << out[i] << ", ";
	//}

	cudaEventRecord(stop_cuda, 0);
	cudaEventSynchronize(stop_cuda);
	cudaEventElapsedTime(&ellapese_time, start_cuda, stop_cuda);
	std::cout << "simulation costs: " << ellapese_time << " ms\n";


	return EXIT_SUCCESS;
}