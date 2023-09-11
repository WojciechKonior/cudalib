#ifndef CUDALIB_H
#define CUDALIB_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>

using namespace std;

typedef char int8;
typedef short int16;
typedef int int32;
typedef long long int64;
typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint32;
typedef unsigned long long uint64;
typedef float float32;
typedef double float64;
typedef long double float128;

namespace cuda
{
    namespace gpu
    {
        template <typename T>
        __global__ void cudaCopyVariableOnGPU(int *i_dest, int *i_src)
        {
            *i_dest = *i_src;
        }

        template <typename T>
        __global__ void cudaCopyVectorOnGPU(int *vec_dest, int *vec_src, size_t n)
        {
            size_t tid = blockDim.x*blockIdx.x + threadIdx.x;
            if(tid<n) vec_dest[tid] = vec_src[tid];
        }

        template <typename T>
        class cudaVariable
        {
        protected:
            T *data;

        public:
            virtual void cudaClear()
            {
                if (data != nullptr)
                {
                    cudaFree(data);
                    data = nullptr;
                }
            }

            virtual void cudaDeclare(size_t size = 1)
            {
                cudaClear();
                cudaMalloc(&this->data, size*sizeof(T));
            }
        };

        template <typename T>
        class scalar : public cudaVariable<T>
        {
        public:
            scalar() { this->cudaDeclare(); }
            scalar(T val) { cudaAssign(val); }
            scalar(const scalar<T> &val) { cudaCopy(val); }
            ~scalar() { this->cudaClear(); }

            T get() const
            {
                T host_var;
                cudaMemcpy(&host_var, this->data, sizeof(T), cudaMemcpyDeviceToHost);
                return host_var;
            }

            operator T() const { return get(); }

            void cudaAssign(T val)
            {
                this->cudaDeclare();
                cudaMemcpy(this->data, &val, sizeof(T), cudaMemcpyHostToDevice);
            }

            void cudaCopy(const scalar<T> &val)
            {
                this->cudaDeclare();
                cudaCopyVariableOnGPU<T><<<1, 1>>>(this->data, val.data);
            }

        };

        typedef scalar<char> int8;
        typedef scalar<short> int16;
        typedef scalar<int> int32;
        typedef scalar<long long> int64;
        typedef scalar<unsigned char> uint8;
        typedef scalar<unsigned short> uint16;
        typedef scalar<unsigned int> uint32;
        typedef scalar<unsigned long long> uint64;
        typedef scalar<float> float32;
        typedef scalar<double> float64;
        typedef scalar<long double> float128;

        template <typename T>
        class vector : public cudaVariable<T>
        {
        protected:
            size_t size;

        public:
            vector() { size = 0; }
            vector(std::vector<T>& vec) { cudaAssign(vec); }
            vector(const cuda::gpu::vector<T> &cuda_vec) { cudaCopy(cuda_vec); }
            ~vector() { this->cudaClear(); }

            std::vector<T> get() const
            {
                std::vector<T> host_vec(size);
                cudaMemcpy(&host_vec[0], this->data, size*sizeof(T), cudaMemcpyDeviceToHost);
                return host_vec;
            }
            
            operator std::vector<T>() const { return get(); }

            void cudaAssign(std::vector<T>& vec)
            {
                this->cudaDeclare();
                cudaMemcpy(this->data, &vec[0], size*sizeof(T), cudaMemcpyHostToDevice);
            }

            void cudaCopy(const cuda::gpu::vector<T> &cuda_vec)
            {
                this->cudaDeclare();
                size_t NUM_THR = size;
                size_t NUM_BLOCKS = 1;
                cudaCopyVectorOnGPU<T><<<NUM_BLOCKS, NUM_THR>>>(this->data, cuda_vec.data, size);
            }

        };
    }
}

template <typename T>
ostream &operator<<(ostream &os, const cuda::gpu::scalar<T> &gpu_val)
{
    os << gpu_val.get();
    return os;
}

template <typename T>
ostream &operator<<(ostream &os, const cuda::gpu::vector<T> &gpu_val)
{
    std::vector<T> result = gpu_val.get();
    os << "[";
    for(auto& r : result)
        os << r << " ";
    os << "]";
    return os;
}

#endif
