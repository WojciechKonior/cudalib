#ifndef VECTOR_TESTS_H
#define VECTOR_TESTS_H

#include <gtest/gtest.h>
#include "../cudavector.cuh"

using namespace cuda;

template<typename T>
class testEmptyVectorInitialization
{
public:
    testEmptyVectorInitialization()
    {
        gpu::vector<T> vec;
        EXPECT_EQ(vec.size(), 0);
        EXPECT_EQ(vec.getptr(), nullptr);
    }
};

TEST(cuda_vector_test, empty_vectors_initialization)
{
    testEmptyVectorInitialization<int8>();
    testEmptyVectorInitialization<int16>();
    testEmptyVectorInitialization<int32>();
    testEmptyVectorInitialization<int64>();

    testEmptyVectorInitialization<uint8>();
    testEmptyVectorInitialization<uint16>();
    testEmptyVectorInitialization<uint32>();
    testEmptyVectorInitialization<uint64>();    

    testEmptyVectorInitialization<float32>();
    testEmptyVectorInitialization<float64>();
}

template<typename T>
class testVectorAssignment
{
public:
    testVectorAssignment()
    {
        std::vector<T> stdVec = {1, 2, 3, 4, 5, 6, 7, 8, 9, 0};

        gpu::vector<T> gpuVec1(stdVec);
        EXPECT_EQ(stdVec, gpuVec1.get());

        gpu::vector<T> gpuVec2 = stdVec;
        EXPECT_EQ(stdVec, gpuVec2.get());

        gpu::vector<T> gpuVec3({1, 2, 3, 4, 5, 6, 7, 8, 9, 0});
        EXPECT_EQ(stdVec, gpuVec3.get());

        gpu::vector<T> gpuVec4 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 0};
        EXPECT_EQ(stdVec, gpuVec4.get());

        stdVec = {1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2};
        gpuVec4 = stdVec;
        EXPECT_EQ(stdVec, gpuVec4.get());

        gpuVec4 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4};
        stdVec.push_back(3);
        stdVec.push_back(4);
        EXPECT_EQ(stdVec, gpuVec4.get());
    }
};

TEST(cuda_vector_test, vectors_assignment)
{
    testVectorAssignment<int8>();
    testVectorAssignment<int16>();
    testVectorAssignment<int32>();
    testVectorAssignment<int64>();

    testVectorAssignment<uint8>();
    testVectorAssignment<uint16>();
    testVectorAssignment<uint32>();
    testVectorAssignment<uint64>();    

    testVectorAssignment<float32>();
    testVectorAssignment<float64>();
}

template<typename T>
class testVectorCopying
{
public:
    testVectorCopying()
    {
        std::vector<T> sample = {7, 8, 4, 3, 2};
        
        gpu::vector<T> a(sample);
        EXPECT_EQ(sample, a.get());

        gpu::vector<T> b(a);
        EXPECT_EQ(sample, b.get());

        gpu::vector<T> c = b;
        EXPECT_EQ(sample, c.get());

        sample.push_back(9);
        a = sample;
        b = a;
        EXPECT_EQ(sample, b.get());
    }
};

TEST(cuda_vector_test, vectors_copying)
{
    testVectorCopying<int8>();
    testVectorCopying<int16>();
    testVectorCopying<int32>();
    testVectorCopying<int64>();

    testVectorCopying<uint8>();
    testVectorCopying<uint16>();
    testVectorCopying<uint32>();
    testVectorCopying<uint64>();    

    testVectorCopying<float32>();
    testVectorCopying<float64>();
}

template<typename T>
class testVectorConverting
{
public:
    testVectorConverting()
    {
        std::vector<T> sample = {2, 3, 1, 5, 3};
        gpu::vector<T> to_gpu(sample);

        std::vector<T> back_from_gpu_1 = to_gpu;
        std::vector<T> back_from_gpu_2(to_gpu);

        EXPECT_EQ(sample, back_from_gpu_1);
        EXPECT_EQ(sample, back_from_gpu_2);
    }
};

TEST(cuda_vector_test, vectors_converting)
{
    testVectorConverting<int8>();
    testVectorConverting<int16>();
    testVectorConverting<int32>();
    testVectorConverting<int64>();

    testVectorConverting<uint8>();
    testVectorConverting<uint16>();
    testVectorConverting<uint32>();
    testVectorConverting<uint64>();    

    testVectorConverting<float32>();
    testVectorConverting<float64>();
}

#endif