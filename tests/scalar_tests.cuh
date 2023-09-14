#ifndef SCALAR_TESTS_H
#define SCALAR_TESTS_H

#include <gtest/gtest.h>
#include "../cudascalar.cuh"

using namespace cuda;

template<typename T>
class testEmptyScalarInitialization
{
public:
    testEmptyScalarInitialization()
    {
        T var;
        EXPECT_EQ(var.get(), 0);
        EXPECT_EQ(var.getptr(), nullptr);
    }
};

TEST(cuda_scalar_test, empty_scalars_initialization)
{
    testEmptyScalarInitialization<gpu::int8>();
    testEmptyScalarInitialization<gpu::int16>();
    testEmptyScalarInitialization<gpu::int32>();
    testEmptyScalarInitialization<gpu::int64>();

    testEmptyScalarInitialization<gpu::uint8>();
    testEmptyScalarInitialization<gpu::uint16>();
    testEmptyScalarInitialization<gpu::uint32>();
    testEmptyScalarInitialization<gpu::uint64>();    

    testEmptyScalarInitialization<gpu::float32>();
    testEmptyScalarInitialization<gpu::float64>();
}

template<typename T>
class testScalarAssignment
{
public:
    testScalarAssignment()
    {
        T a(10);
        EXPECT_EQ(a.get(), 10);
        
        T b = 20;
        EXPECT_EQ(b.get(), 20);

        int cpu_var = 30;
        T c = cpu_var;
        EXPECT_EQ(c.get(), T(cpu_var));

        c = 40;
        EXPECT_EQ(c.get(), 40);
    }
};

TEST(cuda_scalar_test, value_assignments)
{
    testScalarAssignment<gpu::int8>();
    testScalarAssignment<gpu::int16>();
    testScalarAssignment<gpu::int32>();
    testScalarAssignment<gpu::int64>();

    testScalarAssignment<gpu::uint8>();
    testScalarAssignment<gpu::uint16>();
    testScalarAssignment<gpu::uint32>();
    testScalarAssignment<gpu::uint64>();    

    testScalarAssignment<gpu::float32>();
    testScalarAssignment<gpu::float64>();
}

template<typename T>
class testScalarCopying
{
public:
    testScalarCopying()
    {
        T a(10);
        T b(a);
        EXPECT_EQ(a.get(), b.get());
        
        T c = 20;
        T d = c;
        EXPECT_EQ(c.get(), d.get());

        T e(30);
        T &f = e;
        EXPECT_EQ(f.get(), 30);

        e = c;
        EXPECT_EQ(e.get(), 20);
    }
};

TEST(cuda_scalar_test, value_copying)
{
    testScalarCopying<gpu::int8>();
    testScalarCopying<gpu::int16>();
    testScalarCopying<gpu::int32>();
    testScalarCopying<gpu::int64>();

    testScalarCopying<gpu::uint8>();
    testScalarCopying<gpu::uint16>();
    testScalarCopying<gpu::uint32>();
    testScalarCopying<gpu::uint64>();    

    testScalarCopying<gpu::float32>();
    testScalarCopying<gpu::float64>();
}

template<typename T, typename _T>
class testScalarConverting
{
public:
    testScalarConverting()
    {
        _T a = 10;
        T b = a;
        _T c = a;
        EXPECT_EQ(c, a);
    }
};

TEST(cuda_scalar_test, value_converting)
{
    testScalarConverting<gpu::int8, int8>();
    testScalarConverting<gpu::int16, int16>();
    testScalarConverting<gpu::int32, int32>();
    testScalarConverting<gpu::int64, int64>();

    testScalarConverting<gpu::uint8, uint8>();
    testScalarConverting<gpu::uint16, uint16>();
    testScalarConverting<gpu::uint32, uint32>();
    testScalarConverting<gpu::uint64, uint64>();    

    testScalarConverting<gpu::float32, float32>();
    testScalarConverting<gpu::float64, float64>();
}

#endif