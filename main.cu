#include <iostream>
#include "cudascalar.cuh"
#include "cudavector.cuh"

using namespace std;

using namespace cuda;
int main()
{

    // assignTest();
    // copy test
    // output stream test
    // retrieving _data from gpu test

    int32 i = 10;
    gpu::int32 a = i;
    gpu::int32 b = 20;
    gpu::int32 c = a;
    gpu::int32 &e = a;

    std::vector<int32> v = {1, 2, 3, 4};
    gpu::vector<int32> vec8({1, 2, 3, 4});
    gpu::vector<int32> vec9 = {3, 4, 3, 2};
    gpu::vector<int32> vec(v);
    gpu::vector<int32> vec2(vec);
    gpu::vector<int32> vec3 = v;
    gpu::vector<int32> vec4 = vec3;

    std::vector<int32> vec5 = vec4;
    gpu::vector<int32> vec6 = vec5;
    int d = b;
    cout << a << " " << b << " " << c << " " << d << " " << e << endl;
    cout << vec << " "<<vec2 << " " << vec3 << " " << vec4<< " " << vec6 << vec5<< endl;
    return 0;
}