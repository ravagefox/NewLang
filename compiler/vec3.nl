// ---- Vector3 type -----------------------------------------------------------
declare
type struct {
    int x;
    int y;
    int z;

    // canonical constructor
    ctor(x, y, z);
} Vector3;

// ctor body
extend type Vector3:ctor(x, y, z) {
    this->x = x;
    this->y = y;
    this->z = z;
}

// ---- helpers ----------------------------------------------------------------
def sqrt(n) {
    if (n <= 0) {
        finalize 0;
    }
    auto guess = n;
    auto i = 0;
    // a few Newton iterations are enough for our use
    for (i = 0; i < 12; i = i + 1) {
        guess = (guess + n / guess) / 2;
    }
    finalize guess;
}

// ---- instance methods -------------------------------------------------------
extend type Vector3:lengthSq() -> int {
    finalize this->x*this->x + this->y*this->y + this->z*this->z;
}

extend type Vector3:length() -> int {
    auto s = this->x*this->x + this->y*this->y + this->z*this->z;
    finalize sqrt(s);
}

extend type Vector3:dot(other) -> int {
    finalize this->x*other->x + this->y*other->y + this->z*other->z;
}

extend type Vector3:cross(other) -> Vector3 {
    auto cx = this->y*other->z - this->z*other->y;
    auto cy = this->z*other->x - this->x*other->z;
    auto cz = this->x*other->y - this->y*other->x;
    finalize new Vector3(cx, cy, cz);
}

extend type Vector3:add(other) -> Vector3 {
    finalize new Vector3(this->x + other->x,
                         this->y + other->y,
                         this->z + other->z);
}

extend type Vector3:sub(other) -> Vector3 {
    finalize new Vector3(this->x - other->x,
                         this->y - other->y,
                         this->z - other->z);
}

extend type Vector3:scale(s) -> Vector3 {
    finalize new Vector3(this->x * s, this->y * s, this->z * s);
}

extend type Vector3:normalize() -> Vector3 {
    auto len = this->length();
    if (len == 0) {
        finalize new Vector3(0, 0, 0);
    }
    finalize new Vector3(this->x / len, this->y / len, this->z / len);
}
