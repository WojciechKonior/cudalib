#include <iostream>
#include "cudalib.cuh"

using namespace std;

using namespace cuda;
int main()
{

    // assignTest();
    // copy test
    // output stream test
    // retrieving data from gpu test

    int32 i = 10;
    gpu::int32 a = i;
    gpu::int32 b = 20;
    gpu::int32 c = a;

    gpu::vector<int32> vec;
    int d = b;
    cout << a << " " << b << " " << c << " " << d << endl;
    return 0;
}