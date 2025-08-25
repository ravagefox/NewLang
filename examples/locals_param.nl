declare def print(const array<char> s) -> void;
const array<char> helloStr = new("locals ok")->toArray()

__main__ def foo(a, b, c) : void {
    auto<int> x = 3;
    const y = (a + b) / c;
    print(helloStr)
    finalize x + y
}
