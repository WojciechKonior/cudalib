#ifndef CUDALIB_H
#define CUDALIB_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

using namespace std;

template <typename T>
__global__ void cudaCopyVariableOnGPU(int *i_dest, int *i_src)
{
    *i_dest = *i_src;
}

template <typename T>
class cuVar
{
protected:
    T *i;

public:
    cuVar() { cudaDeclare(); }
    cuVar(T val) { cudaAssign(val); }
    cuVar(const cuVar<T> &val) { cudaCopy(val); }
    ~cuVar() { cudaClear(); }

    void cudaClear()
    {
        if (i != nullptr)
        {
            cudaFree(i);
            i = nullptr;
        }
    }

    void cudaDeclare()
    {
        cudaClear();
        cudaMalloc(&i, sizeof(T));
    }

    void cudaAssign(T val)
    {
        cudaDeclare();
        cudaMemcpy(i, &val, sizeof(T), cudaMemcpyHostToDevice);
    }

    void cudaCopy(const cuVar<T> &val)
    {
        cudaDeclare();
        cudaCopyVariableOnGPU<T><<<1, 1>>>(i, val.i);
    }

    T get() const
    {
        T host_var;
        cudaMemcpy(&host_var, i, sizeof(T), cudaMemcpyDeviceToHost);
        return host_var;
    }

    operator T() const { return get(); }
};

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
            T *i;

        public:
            vector(){};
        };

    }
}

template <typename T>
ostream &operator<<(ostream &os, const cuVar<T> &gpu_val)
{
    os << gpu_val.get();
    return os;
}

template <typename T>
ostream &operator<<(ostream &os, const cuda::gpu::vector<T> &gpu_val)
{
    // os << gpu_val.get();
    return os;
}

#endif
