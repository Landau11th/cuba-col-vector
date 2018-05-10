#pragma once

#include <iostream>

#include <cassert>
#include <ctime>

#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>
#include "device_launch_parameters.h"

//define vectors
namespace Deng
{
	namespace CUDA_Vec
	{
		//cublas status
		static cublasHandle_t cublas_handler;
		static cublasStatus_t cublas_status = cublasCreate_v2(&cublas_handler);
		
		class Col;

		//binary operators
		Col operator+(const Col& l_vec, const Col& r_vec);//addition
		Col operator-(const Col& l_vec, const Col& r_vec);//subtraction
		Col operator*(const float& k, const Col& r_vec);//scalar multiplication
		Col operator*(const Col& l_vec, const Col& r_vec);//element-wise vector multiplication

		//curand generator
		static curandGenerator_t gen;
		void rand_seed();//set random seed
		Col rand_normal(const size_t n, const float average = 0.0, const float std_dev = 1.0);//generate normal random sequence

		class Col
		{
		protected:
			size_t _dim;
			float* _vec;//vector of the number field

			//may need a static data member to indicate CUDA runtime error
			//for cudaMemcpy and other functions

		public:
			Col();//default constructor, set _vec to nullptr
			Col(size_t dim);//constructor
			Col(const Col& c);//copy constructor
			Col(Col&& c);//move constructor
			Col(float* vec_host, size_t n);//read data from a host array
			~Col();//destructor. Could be virtual

			//overloading operators as member functions
			Col& operator=(const Col& rhs);//copy assignment operator
			Col& operator=(Col&&rhs) noexcept;//move assignment operator
			//be careful with these operators!!!!!
			Col operator- ();//negation
			void operator+=(const Col& rhs);
			void operator-=(const Col& rhs);
			void operator*=(const float k);//scalar multiplication
			void operator*=(const Col& b);//element-wise multiplication
			
			//member functions
			float* to_host();//copy the data in the device array to a host array
			void set_size(size_t dim);//change the size of the column vector
			size_t size() const { return _dim; };//returns _dim, size of the column vector
			void zeros();//set the elements to 0
			
			friend Col operator+(const Col& l_vec, const Col& r_vec);//addition
			friend Col operator-(const Col& l_vec, const Col& r_vec);//subtraction
			friend Col operator*(const float& k, const Col& r_vec);//scalar multiplication
			friend Col rand_normal(const size_t, const float average, const float std_dev);
			friend Col operator*(const Col& l_vec, const Col& r_vec);//element-wise vector multiplication

		};



	}
}
