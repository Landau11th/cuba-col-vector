#include"cublas_wrapper.h"

using namespace Deng::CUDA_Vec;

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
Col::~Col()
{
	//std::cout << "release memory at " << _vec << std::endl;
	cudaFree(_vec);
	_vec = nullptr;
}
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
	set_zeros<<<2, 512>>> (this->_vec, this->_dim);
}
//need modification
void Col::operator+=(const Col& b)
{

}
void Col::operator-=(const Col& b)
{

}
void Col::operator*=(const float k)
{

}
void Col::operator*=(const Col& b)
{

}




