#ifndef CUDASCALAR_H
#define CUDASCALAR_H

#include "cuda_variable.cuh"
#include <iostream>

namespace cuda
{
    namespace gpu
    {
        template <typename T>
        class scalar : public cudaVariable<T>
        {
        public:
            scalar() { this->_data = nullptr; }
            scalar(T val) 
            { 
                this->cudaDeclare();
                cudaMemcpy(this->_data, &val, sizeof(T), cudaMemcpyHostToDevice);
            }

            scalar(const scalar<T> &val) 
            {
                this->cudaDeclare();
                cudaCopyVariableInGPU<T><<<1, 1>>>(this->_data, val._data);
            }

            ~scalar() { this->cudaClear(); }

            T get() const
            {
                T host_var = 0;
                if(this->_data != nullptr)
                    cudaMemcpy(&host_var, this->_data, sizeof(T), cudaMemcpyDeviceToHost);
                return host_var;
            }

            T* getptr() const { return this->_data; }

            operator T() const { return get(); }
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
    }
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const cuda::gpu::scalar<T> &gpu_val)
{
    os << gpu_val.get();
    return os;
}

#endif
