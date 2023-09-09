#include <iostream>
#include "cudalib.cuh"

using namespace std;


class int32
{
public:
    int32(){cudaDeclare();}
    int32(int value){
        cudaAssign(value);
        cout << "assignment constructor\n";
    }
    ~int32(){ cudaClear(); }

    void cudaClear()
    {
        if(i!=nullptr)
        {
            cudaFree(i);
            i = nullptr;
        }
    }

    void cudaDeclare()
    {
        cudaClear();
        cudaMalloc(&i, sizeof(int));
    }

    void cudaAssign(int value)
    {
        // cudaClear();
        // cudaMalloc(&i, sizeof(int));
        cudaDeclare();
        cudaMemcpy(i, &value, sizeof(int), cudaMemcpyHostToDevice);
    }

    int get()
    {
        int value;
        cudaMemcpy(&value, i, sizeof(int), cudaMemcpyDeviceToHost);
        return value;
    }

public:
    int *i;
};

void assignTest(){
    int a = 10;
    int32 gpu_a = a;
    int b = gpu_a.get();
    if(a == b) cout << "[   OK   ]\n";
    else  cout << "[ NOT OK ]";

    int c = 20;
    int32 gpu_b = c;
    int d = gpu_b.get();
    if(c == d) cout << "[   OK   ]\n";
    else  cout << "[ NOT OK ]";
}

int main()
{

    // assignTest();
    // copy test
    // output stream test
    // retrieving data from gpu test

    cu::int32 a = 10;
    cu::int32 b = 20;
    cu::int32 c = a;
    int d = b;
    cout << a << " " << b << " " << c << " " << d << endl;
    return 0;
}