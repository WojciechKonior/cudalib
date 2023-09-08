#ifndef CUDALIB_H
#define CUDALIB_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
using namespace std;


namespace cu
{
    template<typename T>
    __global__ void cudaCopyVariableOnGPU(int* i_dest, int* i_src)
    {
        *i_dest = *i_src;
    }

    template<typename T>
    class cuVar
    {
    protected:
        T *i;

    public:
        cuVar()
        {
            cudaDeclare(i);
            std::cout << i << " cint default constructor\n";
        }
        cuVar(T val)
        {
            cudaAssign(i, val);
            std::cout << i << " cint assign constructor\n";
        }
        cuVar(const cuVar<T> &val)
        {
            cudaCopy(val);
            std::cout << i << " " << val.i <<" cint cpy constructor\n";
        }
        ~cuVar()
        {
            cudaClear(i);
        }

        void cudaClear(T *var)
        {
            if(var != nullptr)
            {
                // cudaFree(var);
                // var = nullptr;
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

        void cudaCopy(const cuVar<T> &val)
        {
            cout << "cudaCopy: i = " << get() << ", val = " << val.get() << endl;
            cudaDeclare(i);
            cudaCopyVariableOnGPU<T><<<1,1>>>(i, val.i);
        }

        T get() const
        {
            T host_var;
            cudaMemcpy(&host_var, i, sizeof(T), cudaMemcpyDeviceToHost);
            return host_var;
        }
    };

    using int8 = cuVar<char>;
    using int16 = cuVar<short>;
    using int32 = cuVar<int>;
    using int64 = cuVar<long long>;
    using uint8 = cuVar<unsigned char>;
    using uint16 = cuVar<unsigned short>;
    using uint32 = cuVar<unsigned int>;
    using uint64 = cuVar<unsigned long long>;
    using float32 = cuVar<float>;
    using float64 = cuVar<double>;
    using float128 = cuVar<long double>;
}

#endif
