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
        class cuVar
        {
        protected:
            T *data;

        public:
            cuVar() { cudaDeclare(); }
            cuVar(T val) { cudaAssign(val); }
            cuVar(const cuVar<T> &val) { cudaCopy(val); }
            ~cuVar() { cudaClear(); }

            void cudaClear()
            {
                if (data != nullptr)
                {
                    cudaFree(data);
                    data = nullptr;
                }
            }

            void cudaDeclare()
            {
                cudaClear();
                cudaMalloc(&data, sizeof(T));
            }

            void cudaAssign(T val)
            {
                cudaDeclare();
                cudaMemcpy(data, &val, sizeof(T), cudaMemcpyHostToDevice);
            }

            void cudaCopy(const cuVar<T> &val)
            {
                cudaDeclare();
                cudaCopyVariableOnGPU<T><<<1, 1>>>(data, val.data);
            }

            T get() const
            {
                T host_var;
                cudaMemcpy(&host_var, data, sizeof(T), cudaMemcpyDeviceToHost);
                return host_var;
            }

            operator T() const { return get(); }
        };

        typedef cuVar<char> int8;
        typedef cuVar<short> int16;
        typedef cuVar<int> int32;
        typedef cuVar<long long> int64;
        typedef cuVar<unsigned char> uint8;
        typedef cuVar<unsigned short> uint16;
        typedef cuVar<unsigned int> uint32;
        typedef cuVar<unsigned long long> uint64;
        typedef cuVar<float> float32;
        typedef cuVar<double> float64;
        typedef cuVar<long double> float128;

        template <typename T>
        class vector
        {
        protected:
            T *data;
            size_t size;

        public:
            vector() { size = 0; }
            vector(std::vector<T>& vec) { cudaAssign(vec); }
            vector(const cuda::gpu::vector<T> &cuda_vec) { cudaCopy(cuda_vec); }
            ~vector() { cudaClear(); }

            void cudaClear()
            {
                if (data != nullptr)
                {
                    cudaFree(data);
                    data = nullptr;
                }
            }

            void cudaDeclare()
            {
                cudaClear();
                cudaMalloc(&data, size*sizeof(T));
            }

            void cudaAssign(std::vector<T>& vec)
            {
                cudaDeclare();
                cudaMemcpy(data, &vec[0], size*sizeof(T), cudaMemcpyHostToDevice);
            }

            void cudaCopy(const cuda::gpu::vector<T> &cuda_vec)
            {
                cudaDeclare();
                size_t NUM_THR = size;
                size_t NUM_BLOCKS = 1;
                cudaCopyVectorOnGPU<T><<<NUM_BLOCKS, NUM_THR>>>(data, cuda_vec.data, size);
            }

            T get() const
            {
                std::vector<T> host_vec(size);
                cudaMemcpy(&host_vec[0], data, size*sizeof(T), cudaMemcpyDeviceToHost);
                return host_vec;
            }

            operator T() const { return get(); }
        };
    }
}

template <typename T>
ostream &operator<<(ostream &os, const cuda::gpu::cuVar<T> &gpu_val)
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
