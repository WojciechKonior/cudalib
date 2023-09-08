#include <iostream>
#include "cudalib.cuh"

using namespace std;

int main()
{
    cu::int32 a = 10;
    cu::int32 b = 20;
    // b = a;
    cout << "the result = " << a.get() << " " << b.get() << endl;
    cout << "Hello World\n";
    return 0;
}