// std.nl
import { "vec3.nl" };

declare
type struct {
    
} int;
declare
type struct { 

} char;
declare
type struct { 

} bool;
declare
type struct { 

} void;
declare
type struct { 
    
} array;

declare def print(const array<char> s) -> void;
declare def println(const array<char> s) -> void;
declare def toArray(const array<char> s) -> array<char>;


// --------------- Array STD ---------------
// append at end: size -> size+1
declare def append(array numbers, int size, value) -> array;
def append(numbers, size, value) : array {
    auto copy = new[size + 1];

    auto i = 0;
    while (1) {
        end i >= size;
        copy[i] = numbers[i];
        i = i + 1;
    }

    copy[size] = value;
    finalize copy;
}

// insert at index: size -> size+1
declare def insert(array numbers, int size, int idx, value) -> array;
def insert(numbers, size, idx, value) : array {
    auto copy = new[size + 1];

    // copy [0..idx-1]
    auto i = 0;
    while (1) {
        end i >= idx;
        copy[i] = numbers[i];
        i = i + 1;
    }

    // place new value
    copy[idx] = value;

    // shift/copy the rest
    auto j = idx;
    while (1) {
        end j >= size;
        copy[j + 1] = numbers[j];
        j = j + 1;
    }

    finalize copy;
}

// pop last: size -> size-1 (drops last element)
declare def pop(array numbers, int size) -> array;
def pop(numbers, size) : array {
    // guard: if size <= 0, return empty same-size (or choose your convention)
    auto newSize = size - 1;
    if newSize < 0 { finalize numbers; }

    auto copy = new[newSize];

    auto i = 0;
    while (1) {
        end i >= newSize;
        copy[i] = numbers[i];
        i = i + 1;
    }

    finalize copy;
}

// remove-at index: size -> size-1
declare def remove_at(array numbers, int size, int idx) -> array;
def remove_at(numbers, size, idx) : array {
    auto copy = new[size - 1];

    // copy [0..idx-1]
    auto i = 0;
    while (1) {
        end i >= idx;
        copy[i] = numbers[i];
        i = i + 1;
    }

    // copy [idx+1..size-1] shifted left by 1
    auto j = idx + 1;
    while (1) {
        end j >= size;
        copy[j - 1] = numbers[j];
        j = j + 1;
    }

    finalize copy;
}
