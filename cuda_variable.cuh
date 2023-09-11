#ifndef CUDAVARIABLE_H
#define CUDAVARIABLE_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

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
        class cudaVariable
        {
        protected:
            T *_data;

        public:
            virtual void cudaClear()
            {
                if (_data != nullptr)
                {
                    cudaFree(_data);
                    _data = nullptr;
                }
            }

            virtual void cudaDeclare(size_t size = 1)
            {
                cudaClear();
                cudaMalloc(&this->_data, size*sizeof(T));
            }
        };
    }
}

#endif
