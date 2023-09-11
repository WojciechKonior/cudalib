#ifndef CUDAVECTOR_H
#define CUDAVECTOR_H

#include "cuda_variable.cuh"
#include <initializer_list>
#include <iostream>
#include <vector>

namespace cuda
{
    namespace gpu
    {
        template <typename T>
        __global__ void cudaCopyVectorOnGPU(int *vec_dest, int *vec_src, size_t n)
        {
            size_t tid = blockDim.x*blockIdx.x + threadIdx.x;
            if(tid<n) vec_dest[tid] = vec_src[tid];
        }

        template <typename T>
        class vector : public cudaVariable<T>
        {
        protected:
            size_t _size;

        public:
            vector() { _size = 0; }
            vector(std::vector<T>& vec) { cudaAssign(vec); }
            vector(const std::initializer_list<T>& list) { std::vector<T> vec = list; cudaAssign(vec); }
            vector(cuda::gpu::vector<T> &cuda_vec) { cudaCopy(cuda_vec); }
            ~vector() { this->cudaClear(); }

            std::vector<T> get() const
            {
                std::vector<T> host_vec(_size);
                cudaMemcpy(&host_vec[0], this->_data, _size*sizeof(T), cudaMemcpyDeviceToHost);
                return host_vec;
            }
            
            operator std::vector<T>() const { return get(); }

            void cudaAssign(std::vector<T>& vec)
            {
                _size = vec.size();
                this->cudaDeclare(_size);
                cudaMemcpy(this->_data, &vec[0], _size*sizeof(T), cudaMemcpyHostToDevice);
            }

            void cudaAssign(std::initializer_list<T>& list)
            {
                std::vector<T> vec = list;
                _size = vec.size();
                this->cudaDeclare(_size);
                cudaMemcpy(this->_data, &vec[0], _size*sizeof(T), cudaMemcpyHostToDevice);
            }

            void cudaCopy(cuda::gpu::vector<T> &cuda_vec)
            {
                _size = cuda_vec.size();
                this->cudaDeclare(_size);
                size_t NUM_THR = _size;
                size_t NUM_BLOCKS = 1;
                cudaCopyVectorOnGPU<T><<<NUM_BLOCKS, NUM_THR>>>(this->_data, cuda_vec._data, _size);
            }

            size_t size(){ return this->_size; }

        };
    }
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const cuda::gpu::vector<T> &gpu_val)
{
    std::vector<T> result = gpu_val.get();
    os << "[";
    for(auto& r : result)
        os << r << " ";
    os << "]";
    return os;
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &vec)
{
    os << "[";
    for(auto& r : vec)
        os << r << " ";
    os << "]";
    return os;
}

#endif
