#include"cublas_wrapper.h"

using namespace Deng::CUDA_Vec;

//explicit instantiations

//template class Col<int>;
template class Col<float>;
//template class Col<double>;
//template class Col<std::complex<float> >;
//template class Col<std::complex<double> >;

//#define DENG_VECTOR_COMPLEX //to determine whether the field is complex (aim for inner product)
template <typename Field>
Col<Field>::Col()//default constructor, set _vec to nullptr
{
	_dim = 0;
	_vec = nullptr;
}
template <typename Field>
Col<Field>::Col(size_t dim)
{
	_dim = dim;
	cudaMalloc((void **)& _vec, _dim * sizeof(Field));
}
template <typename Field>
Col<Field>::Col(const Col<Field>& c)//copy constructor
{
	_dim = c.size();
	cudaMalloc((void **)& _vec, _dim * sizeof(Field));
	cublasScopy(cublas_handler, _dim, c._vec, 1, _vec, 1);
}
template <typename Field>
Col<Field>::Col(Col<Field>&& c)//move constructor
{
	_dim = c.size();
	_vec = c._vec;
	c._vec = nullptr;
}
template <typename Field>
void Col<Field>::set_size(size_t dim)
{
	//if the size is already dim, there is no need to change
	if (_dim != dim)
	{
		_dim = dim;
		cudaFree(_vec);
		cudaMalloc((void **)& _vec, _dim * sizeof(Field));
	}
}
template <typename Field>
Col<Field>::~Col()
{
	//std::cout << "release memory at " << _vec << std::endl;
	cudaFree(_vec);
	_vec = nullptr;
}
template <typename Field>
Col<Field>& Col<Field>::operator=(const Col<Field> & b)
{
	set_size(b.size());
	cublasScopy(cublas_handler, _dim, b._vec, 1, _vec, 1);
	return *this;
}
template <typename Field>
Col<Field>& Col<Field>::operator=(Col<Field> &&rhs) noexcept
{
	assert((this != &rhs) && "Memory clashes in operator '=&&'!\n");
	cudaFree(this->_vec);
	this->_vec = rhs._vec;
	this->_dim = rhs.size();
	rhs._vec = nullptr;
	return *this;
}
//need modification
template <typename Field>
void Col<Field>::operator+=(const Col<Field>& b)
{
	for (unsigned int i = 0; i < this->_dim; ++i)
	{
		this->_vec[i] += b[i];
	}
}
template <typename Field>
void Col<Field>::operator-=(const Col<Field>& b)
{
	for (unsigned int i = 0; i < this->_dim; ++i)
	{
		this->_vec[i] -= b[i];
	}
}
template <typename Field>
void Col<Field>::operator*=(const Field k)
{
	//here neglect all the size check etc
	//since +=, -=, *= are usually used for optimal performance
	for (unsigned int i = 0; i < this->_dim; ++i)
	{
		this->_vec[i] *= k;
	}
}
template <typename Field>
void Col<Field>::operator*=(const Col<Field>& b)
{
	//here neglect all the size check etc
	//since +=, -=, *= are usually used for optimal performance
	for (unsigned int i = 0; i < this->_dim; ++i)
	{
		this->_vec[i] *= b[i];
	}
}
template <typename Field>
Field dot_product(Col<Field> a, Col<Field> b)
{
	int dim = a.size();
	Field dot;

	if (dim == b.size())
	{
		int i;

		dot = 0.0;

		for (i = 0; i<dim; i++)
		{
			dot += a[i] * b[i];
		}
	}
	else
	{
		printf("Error in dot product!\n");
	}

	return dot;
}
