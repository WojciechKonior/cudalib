#include "tests/scalar_tests.cuh"
#include "tests/vector_tests.cuh"

int main(int argc, char **argv)
{
    
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();

    // std::vector<int> v0 = {6, 7, 8, 9, 0};
    // gpu::vector<int> v1 = {1, 2, 3, 4, 5};
    
    // std::cout << v1 << std::endl;

    // v1 = v0;
    // std::cout << v1.get() << std::endl;


    // return 0;
}