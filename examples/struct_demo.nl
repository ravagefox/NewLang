

const array<char> helloWorld = std:toArray("HelloWorld");

declare
type struct {
    int param1;
    const int param2 = 0;
    private int protectedVariable;
    
    ctor(param1);
} SomeClass1;

declare
type struct from SomeClass1 {
    array numbers;
} SomeClass2;

extend type SomeClass1:ctor(param1, param2, param3) into base(param1) {
    this->param1 = param1;
    this->protectedVariable = 0;
    // finalize optional; ignored by executor
}

extend type SomeClass1:getParam() -> int {
    finalize this->param1;
}

__main__ 
def foo() : void {
    Vector3 a(3, 4, 0);
    Vector3 b(1, 2, 3);

    std:println("a");              std:println(a);          // prints dict-like object
    std:println("b");              std:println(b);

    auto lenA = a->length();       std:println(lenA);       // 5 (approx)
    auto dotAB = a->dot(b);        std:println(dotAB);      // 3*1 + 4*2 + 0*3 = 11
    auto crossAB = a->cross(b);    std:println(crossAB);    // (-12, 9, 2)

    auto sum  = a->add(b);         std:println(sum);        // (4, 6, 3)
    auto diff = a->sub(b);         std:println(diff);       // (2, 2, -3)
    auto s    = a->scale(2.5);     std:println(s);          // (7.5, 10, 0)
    auto nrm  = a->normalize();    std:println(nrm);        // (~0.6, ~0.8, 0)

}
