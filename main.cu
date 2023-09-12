#include "tests/scalar_tests.cuh"
#include "tests/vector_tests.cuh"

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}