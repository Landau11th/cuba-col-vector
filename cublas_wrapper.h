#pragma once

#include <iostream>

#include<cassert>

#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

//define vectors
namespace Deng
{
	namespace CUDA_Vec
	{

		static cublasHandle_t cublas_handler;
		static cublasStatus_t cublas_status = cublasCreate_v2(&cublas_handler);
		
		class Col;

		//addition
		Col operator+(const Col& l_vec, const Col& r_vec);
		//subtraction
		Col operator-(const Col& l_vec, const Col& r_vec);
		//scalar multiplication
		Col operator*(const float& k, const Col& r_vec);
		

		class Col
		{
		protected:
			size_t _dim;
			float* _vec;//vector of the number field
		public:
			Col();//default constructor, set _vec to nullptr
			Col(size_t dim);//constructor
			Col(const Col& c);//copy constructor
			Col(Col&& c);//move constructor

			//change the size of the column vector
			void set_size(size_t dim);
			//returns _dim, size fo the column vector
			size_t size() const { return _dim; };
			//set the vector to be 0
			void zeros();
			//destructor. Could be virtual
			//virtual
			~Col();

			//overloading operators
			//member functions
			//copy assignment operator
			Col& operator=(const Col& rhs);
			//move constructor
			Col& operator=(Col&&rhs) noexcept;
			//other assignment operator
			//be careful with these operators!!!!!
			void operator+=(const Col& rhs);
			void operator-=(const Col& rhs);
			void operator*=(const float k);//scalar multiplication
			void operator*=(const Col& b);//element-wise multiplication
														//element access

			//non-member operator
			//negative
			Col& operator- ()
			{
				Col a(this->size());
				a.zeros();

				return a;
			}

			//addition
			friend Col operator+(const Col& l_vec, const Col& r_vec);
			//subtraction
			friend Col operator-(const Col& l_vec, const Col& r_vec);
			//scalar multiplication
			friend Col operator*(const float& k, const Col& r_vec);			

		};


		const float one = 1.0;
		//addition
		inline Col operator+(const Col& l_vec, const Col& r_vec)//addition
		{
			size_t dim = l_vec.size();
			assert((dim == r_vec.size()) && "Dimension mismatch in + (vector addition)!");

			auto a = r_vec;
			cublasSaxpy(cublas_handler, dim, &one,
				l_vec._vec, 1,
				a._vec, 1);

			return a;
		}
		//subtraction
		inline Col operator-(const Col& l_vec, const Col& r_vec)//subtraction
		{
			const size_t dim = l_vec.size();
			assert((dim == r_vec.size()) && "Dimension mismatch in - (vector subtraction)!");

			auto a = r_vec;
			a = -a;
			cublasSaxpy(cublas_handler, dim, &one,
				l_vec._vec, 1,
				a._vec, 1);

			return a;
		}
		//scalar multiplication
		inline Col operator*(const float& k, const Col& r_vec)//scalar multiplication
		{
			const size_t dim = r_vec.size();
			Col a(dim);
			a.zeros();

			cublasSaxpy(cublas_handler, dim, &k,
				r_vec._vec, 1,
				a._vec, 1);

			return a;
		}

	}
}
