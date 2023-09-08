#ifndef CUDALIB_H
#define CUDALIB_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

// #define cu::char cvar<char>
// #define cu::short cvar<short>
#define cuInt cuVar<int>
// #define cu::long cvar<long long>
// #define cu::uchar cvar<unsigned char>
// #define cu::ushort cvar<unsigned short>
// #define cu::uint cvar<unsigned int>
// #define cu::ulong cvar<unsigned long long>
// #define cu::float cvar<float>
// #define cu::double cvar<double>
// #define cu::ldouble cvar<ldouble>

template<typename T>
class cuVar
{
protected:
    T *i = nullptr;

public:
    cuVar()
    {
        std::cout << "cint default constructor\n";
        cudaDeclare(i);
    }
    cuVar(T val)
    {
        std::cout << "cint val constructor\n";
        cudaAssign(i, val);
    }
    cuVar(const cuVar<T> &val)
    {
        std::cout << "cint cpy constructor\n";
        cudaReference(val);
    }
    ~cuVar()
    {
        cudaClear(i);
    }

    void cudaClear(T *var)
    {
        if(var != nullptr)
        {
            cudaFree(var);
            var = nullptr;
        }
    }

    void cudaDeclare(T *var)
    {
        cudaClear(var);
        cudaMalloc(&var, sizeof(T));
    }

    void cudaAssign(T *ptr, T val)
    {
        cudaDeclare(ptr);
        cudaMemcpy(ptr, &val, sizeof(T), cudaMemcpyHostToDevice);
    }

    void cudaReference(const cuVar<T> &val)
    {
        cudaClear(i);
        i = val.i;
    }
};

#endif
