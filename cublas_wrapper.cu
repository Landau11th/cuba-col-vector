#include"cublas_wrapper.h"

namespace Deng 
{
	namespace CUDA_Vec 
	{
		//friend operators
		const float one = 1.0;
		const float minus_one = -1.0;

		const size_t num_SM = 3;
		const size_t threads_per_block = 512 * 4;
		
		//////////////////////////////////
		//binary operators
		Col operator+(const Col& l_vec, const Col& r_vec)//addition
		{
			size_t dim = l_vec.size();
			assert((dim == r_vec.size()) && "Dimension mismatch in + (vector addition)!");

			auto a = l_vec;
			cublasSaxpy(cublas_handler, dim, &one, r_vec._vec, 1, a._vec, 1);

			return a;
		}
		Col operator-(const Col& l_vec, const Col& r_vec)//subtraction
		{
			const size_t dim = l_vec.size();
			assert((dim == r_vec.size()) && "Dimension mismatch in - (vector subtraction)!");

			Col a = l_vec;
			cublasSaxpy(cublas_handler, dim, &minus_one, r_vec._vec, 1, a._vec, 1);

			return a;
		}
		Col operator*(const float& k, const Col& r_vec)//scalar multiplication
		{
			const size_t dim = r_vec.size();
			auto a = r_vec;
			cublasSscal(cublas_handler, dim, &k, a._vec, 1);
			return a;
		}
		__global__ void element_wise_multiplication(float* left, float* right, const size_t length)//overwrite the values in right
		{
			size_t thrd_id = threadIdx.x + blockIdx.x * blockDim.x;
			while (thrd_id < length)
			{
				right[thrd_id] *= left[thrd_id];
				thrd_id += blockDim.x* gridDim.x;
			}
		}
		Col operator*(const Col& l_vec, const Col& r_vec)//element-wise vector multiplication
		{
			const size_t dim = l_vec.size();
			assert((dim == r_vec.size()) && "Dimension mismatch in - (vector subtraction)!");
			auto a = r_vec;
			element_wise_multiplication <<< num_SM, threads_per_block >>> (l_vec._vec, a._vec, dim);
			return a;
		}
		
		///////////////////////////////////////
		//generate random sequence
		void rand_seed()
		{
			curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
			time_t seed;
			time(&seed);
			curandSetPseudoRandomGeneratorSeed(gen, seed);
		}
		Col rand_normal(const size_t n, const float average, const float std_dev)
		{
			Col a(n);
			curandGenerateNormal(gen, a._vec, n, average, std_dev);
			return a;
		}

		////////////////////////////////////////////
		//constructors and destructors
		//REMINDER: need to implement check on CUDA runtime errors
		Col::Col()//default constructor, set _vec to nullptr
		{
			_dim = 0;
			_vec = nullptr;
		}
		Col::Col(size_t dim)
		{
			_dim = dim;
			cudaMalloc((void **)& _vec, _dim * sizeof(float));
		}
		Col::Col(const Col& c)//copy constructor
		{
			_dim = c.size();
			cudaMalloc((void **)& _vec, _dim * sizeof(float));
			cublasScopy(cublas_handler, _dim, c._vec, 1, _vec, 1);
		}
		Col::Col(Col&& c)//move constructor
		{
			_dim = c.size();
			_vec = c._vec;
			c._vec = nullptr;
		}
		Col::Col(float* vec_host, size_t n)
		{
			_dim = n;
			cudaMalloc((void **)& _vec, _dim * sizeof(float));
			cudaMemcpy(_vec, vec_host, _dim * sizeof(float), cudaMemcpyHostToDevice);
		}
		Col::~Col()
		{
			//std::cout << "release memory at " << _vec << std::endl;
			cudaFree(_vec);
			_vec = nullptr;
		}

		///////////////////////////////////////////////
		//member operators
		Col& Col::operator=(const Col& b)
		{
			set_size(b.size());
			cublasScopy(cublas_handler, _dim, b._vec, 1, _vec, 1);
			return *this;
		}
		Col& Col::operator=(Col&&rhs) noexcept
		{
			assert((this != &rhs) && "Memory clashes in operator '=&&'!\n");
			cudaFree(this->_vec);
			this->_vec = rhs._vec;
			this->_dim = rhs.size();
			rhs._vec = nullptr;
			return *this;
		}
		//these operators does not check boundaries
		Col Col::operator- ()
		{
			Col a(this->size());
			a.zeros();
			a = a - *this;
			return a;
		}
		void Col::operator+=(const Col& r_vec)
		{
			cublasSaxpy(cublas_handler, _dim, &one, r_vec._vec, 1, _vec, 1);
		}
		void Col::operator-=(const Col& r_vec)
		{
			cublasSaxpy(cublas_handler, _dim, &minus_one, r_vec._vec, 1, _vec, 1);
		}
		void Col::operator*=(const float k)
		{
			cublasSscal(cublas_handler, _dim, &k, _vec, 1);
		}
		void Col::operator*=(const Col& r_vec)
		{
			element_wise_multiplication <<< num_SM, threads_per_block >>> (r_vec._vec, _vec, _dim);
		}

		///////////////////////////////////////////////
		//member functions
		float* Col::to_host()
		{
			float* vec_host = new float[_dim];
			cudaMemcpy(vec_host, _vec, _dim * sizeof(float), cudaMemcpyDeviceToHost);
			return vec_host;
		}
		void Col::set_size(size_t dim)
		{
			//if the size is already dim, there is no need to change
			if (_dim != dim)
			{
				_dim = dim;
				cudaFree(_vec);
				cudaMalloc((void **)& _vec, _dim * sizeof(float));
			}
		}
		__global__ void set_zeros(float* x_device, const size_t length)
		{
			size_t thrd_id = threadIdx.x + blockIdx.x * blockDim.x;

			while (thrd_id < length)
			{
				x_device[thrd_id] = 0.0f;
				thrd_id += blockDim.x* gridDim.x;
			}
		}
		void Col::zeros()
		{
			set_zeros <<< num_SM, threads_per_block >>> (this->_vec, this->_dim);
		}


	}
}