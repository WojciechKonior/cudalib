#include <iostream>
#include "cudascalar.cuh"
#include "cudavector.cuh"
#include <gtest/gtest.h>

// using namespace std;

using namespace cuda;

TEST(first_test, f_test)
{
    gpu::int32 a;
    EXPECT_EQ(a.get(), 0);
    EXPECT_EQ(a.getptr(), nullptr);
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();

    // // assignTest();
    // // copy test
    // // output stream test
    // // retrieving _data from gpu test

    // gpu::int32
    // int32 i = 10;
    // gpu::int32 a = i;
    // gpu::int32 b = 20;
    // gpu::int32 c = a;
    // gpu::int32 &e = a;

    // std::vector<int32> v = {1, 2, 3, 4};
    // gpu::vector<int32> vec8({1, 2, 3, 4});
    // gpu::vector<int32> vec9 = {3, 4, 3, 2};
    // gpu::vector<int32> vec(v);
    // gpu::vector<int32> vec2(vec);
    // gpu::vector<int32> vec3 = v;
    // gpu::vector<int32> vec4 = vec3;

    // std::vector<int32> vec5 = vec4;
    // gpu::vector<int32> vec6 = vec5;
    // int d = b;
    // std::cout << a << " " << b << " " << c << " " << d << " " << e << std::endl;
    // std::cout << vec << " "<<vec2 << " " << vec3 << " " << vec4<< " " << vec6 << vec5<< std::endl;
    // return 0;
}