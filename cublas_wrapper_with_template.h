#pragma once

#include <iostream>

#include<cassert>

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
		
		template <typename Field>
		class Col;

		//addition
		template<typename Field_l, typename Field_r>
		Col<Field_r> operator+(const Col<Field_l>& l_vec, const Col<Field_r> & r_vec)//addition
		{
			size_t dim = l_vec.size();
			assert((dim == r_vec.size()) && "Dimension mismatch in + (vector addition)!");

			auto a = r_vec;
			Field_l one = 1.0;
			cublasSaxpy(cublas_handler, dim, &one,
				l_vec._vec, 1,
				a._vec, 1);

			return a;
		}
		//subtraction
		template<typename Field_l, typename Field_r>
		Col<Field_r> operator-(const Col<Field_l>& l_vec, const Col<Field_r> & r_vec)//subtraction
		{
			const unsigned int dim = l_vec.dimension();
			assert((dim == r_vec.dimension()) && "Dimension mismatch in - (vector subtraction)!");

			Col<Field_r> a(dim);
			for (unsigned int i = 0; i < dim; ++i)
			{
				a[i] = l_vec[i] - r_vec[i];
			}
			return a;
		}
		//scalar multiplication
		template<typename Scalar, typename Field_prime>
		Col<Field_prime> operator*(const Scalar& k, const Col<Field_prime> & r_vec)//scalar multiplication
		{
			const unsigned int dim = r_vec.dimension();
			Col<Field_prime> a(dim);

			for (unsigned int i = 0; i < dim; ++i)
			{
				a[i] = k*r_vec[i];
			}
			return a;
		}
		//element-wise multiplication
		template<typename Field_l, typename Field_r>
		Col<Field_r> operator%(const Col<Field_l>& l_vec, const Col<Field_r> & r_vec)//element-wise multiplication
		{
			const unsigned int dim = l_vec.dimension();
			assert((dim == r_vec.dimension()) && "Dimension mismatch in % (element-wise multiplication)!");

			Col<Field_r> a(dim);
			for (unsigned int i = 0; i < dim; ++i)
			{
				a[i] = l_vec[i] * r_vec[i];
			}
			return a;
		}
		//dot product. Field_l could be either Field_r or Scalar
		//for now only works for real/hermitian matrix!!!!!!!!!!!!!!!
		//choosing ^ is not quite appropriate
		template<typename Field_l, typename Field_r>
		Field_r operator^(const Col<Field_l>& l_vec, const Col<Field_r> & r_vec)
		{
			const unsigned int dim = l_vec.dimension();
			assert((dim == r_vec.dimension()) && "Dimension mismatch in ^ (inner product)!");

			Field_r a = r_vec[0];
			//only work for scalars and matrices
			a = 0 * a;

			for (unsigned int i = 0; i < dim; ++i)
			{
				a += l_vec[i] * r_vec[i];
			}
			return a;
		}

		template <typename Field>
		class Col
		{
		protected:
			size_t _dim;
			Field* _vec;//vector of the number field
		public:
			Col();//default constructor, set _vec to nullptr
			Col(size_t dim);//constructor
			Col(const Col<Field>& c);//copy constructor
			Col(Col<Field>&& c);//move constructor
			
			//change the size of the column vector
			void set_size(size_t dim);
			//returns _dim, size fo the column vector
			size_t size() const { return _dim; };
			//destructor. Could be virtual
			//virtual
			~Col();

			//overloading operators
			//member functions
			//copy assignment operator
			Col<Field> &operator=(const Col<Field> & rhs);
			//move constructor
			Col<Field> &operator=(Col<Field> &&rhs) noexcept;
			//other assignment operator
			//be careful with these operators!!!!!
			void operator+=(const Col<Field>& rhs);
			void operator-=(const Col<Field>& rhs);
			void operator*=(const Field k);//scalar multiplication
			void operator*=(const Col<Field>& b);//element-wise multiplication
														//element access

			//non-member operator
			//negative
			Col<Field>& operator- ()
			{
				Col<Field> a(this->size());
				a.zeros();

				return a;
			}
			//addition
			template<typename Field_l, typename Field_r>
			friend Col<Field_r> operator+(const Col<Field_l>& l_vec, const Col<Field_r> & r_vec);
			//subtraction
			template<typename Field_l, typename Field_r>
			friend Col<Field_r> operator-(const Col<Field_l>& l_vec, const Col<Field_r> & r_vec);
			//scalar multiplication
			template<typename Scalar, typename Field_prime>
			friend Col<Field_prime> operator*(const Scalar & k, const Col<Field_prime> & r_vec);
			//scalar multiplication in another order
			//ambiguous!!!!!!
			//template<typename Scalar, typename Field_prime>
			//friend Col<Field_prime> operator*(const Col<Field_prime> & r_vec, const Scalar & k);
			//element-wise multiplication
			template<typename Field_l, typename Field_r>
			friend Col<Field_r> operator%(const Col<Field_l>& l_vec, const Col<Field_r> & r_vec);
			//inner product. Field_l could be either Field_r or Scalar
			//for now only works for real matrix!!!!!!!!!!!!!!!
			template<typename Field_l, typename Field_r>
			friend Field_r operator^(const Col<Field_l>& l_vec, const Col<Field_r> & r_vec);

			

		};
	}
}
