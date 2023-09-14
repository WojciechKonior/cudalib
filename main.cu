#include "tests/scalar_tests.cuh"
#include "tests/vector_tests.cuh"

int main(int argc, char **argv)
{
    
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();

    // std::vector<int> stdVec = {1, 2, 3, 4, 5, 6, 7, 8, 9, 0};
    // std::vector<int> stdVec2 = {9, 4, 3, 2, 5, 6, 7, 8, 1, 0};

    // gpu::vector<int> gpuVec1(stdVec);
    // // gpu::vector<int> gpuVec2(stdVec);

    // gpuVec1 = stdVec2;

    // // gpuVec2 = gpuVec1;

    // std::cout << "res: " << gpuVec1 << std::endl << std::endl;

    gpu::int32 a = 10;
    a = 20;
    // std::cout << "res " << a.get() << std::endl;
    // std::cout << "res2 " << a << std::endl;
    // a = 30;
    // int c = a;
    std::cout << "res2 " << a.get() << std::endl;
    std::cout << "res3 " << a << std::endl;
    // std::cout << "res3 " << a << std::endl;

    // int *dst = nullptr;
    // int *src = a.getptr();
    // cudaMemcpy(dst, src, sizeof(int), cudaMemcpyDeviceToHost);
    // std::cout << "res " << c << std::endl;

    return 0;
}