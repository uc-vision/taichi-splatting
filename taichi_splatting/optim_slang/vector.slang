import "slangpy";

typedef __BuiltinFloatingPointType Float;

struct Vector<T : Float, let D : int> : IArithmetic {
    T[D] data;

    [mutating]
    __init(Vector<T, D> value) {
        [ForceUnroll]
        for (int d = 0; d < D; d++) {
            data[d] = value.data[d];
        }
    }

    __init(int value) {
        [ForceUnroll]
        for (int d = 0; d < D; d++) {
            data[d] = T(value);
        }
    }

    __init(T[D] value) {
        [ForceUnroll]
        for (int d = 0; d < D; d++) {
            data[d] = value[d];
        }
    }

    inline T dot(Vector<T, D> other) {
        T out = T(0);
        [ForceUnroll]
        for (int d = 0; d < D; d++) {
            out += data[d] * other.data[d];
        }
        return out;
    }

    inline T norm() {
        return dot(this);
    }

    static inline Vector<T, D> init(Vector<T, D> value) {
        return Vector<T, D>(value.data);
    }

    inline Vector<T, D> add(Vector<T, D> other) {
        T[D] out;
        [ForceUnroll]
        for (int d = 0; d < D; d++) {
            out[d] = data[d] + other.data[d];
        }
        return Vector<T, D>(out);
    }

    inline Vector<T, D> add(T scalar) {
        T[D] out;
        [ForceUnroll]
        for (int d = 0; d < D; d++) {
            out[d] = data[d] + scalar;
        }
        return Vector<T, D>(out);
    }

    inline Vector<T, D> mul(T scalar) {
        T[D] out;
        [ForceUnroll]
        for (int d = 0; d < D; d++) {
            out[d] = data[d] * scalar;
        }
        return Vector<T, D>(out);
    }

    inline Vector<T, D> mul(Vector<T, D> other) {
        T[D] out;
        [ForceUnroll]
        for (int d = 0; d < D; d++) {
            out[d] = data[d] * other.data[d];
        }
        return Vector<T, D>(out);
    }

    inline Vector<T, D> sub(Vector<T, D> other) {
        T[D] out;
        [ForceUnroll]
        for (int d = 0; d < D; d++) {
            out[d] = data[d] - other.data[d];
        }
        return Vector<T, D>(out);
    }

    inline Vector<T, D> sub(T scalar) {
        T[D] out;
        [ForceUnroll]
        for (int d = 0; d < D; d++) {
            out[d] = data[d] - scalar;
        }
        return Vector<T, D>(out);
    }

    inline Vector<T, D> div(T scalar) {
        T[D] out;
        [ForceUnroll]
        for (int d = 0; d < D; d++) {
            out[d] = data[d] / scalar;
        }
        return Vector<T, D>(out);
    }

    inline Vector<T, D> div(Vector<T, D> other) {
        T[D] out;
        [ForceUnroll]
        for (int d = 0; d < D; d++) {
            out[d] = data[d] / other.data[d];
        }
        return Vector<T, D>(out);
    }

    inline Vector<T, D> neg() {
        T[D] out;
        [ForceUnroll]
        for (int d = 0; d < D; d++) {
            out[d] = -data[d];
        }
        return Vector<T, D>(out);
    }

    inline Vector<T, D> mod(T scalar) {
        T[D] out;
        [ForceUnroll]
        for (int d = 0; d < D; d++) {
            out[d] = data[d] % scalar;
        }
        return Vector<T, D>(out);
    }

    inline Vector<T, D> mod(Vector<T, D> other) {
        T[D] out;
        [ForceUnroll]
        for (int d = 0; d < D; d++) {
            out[d] = data[d] % other.data[d];
        }
        return Vector<T, D>(out);
    }

    inline bool equals(Vector<T, D> other) {
        [ForceUnroll]
        for (int d = 0; d < D; d++) {
            if (data[d] != other.data[d]) return false;
        }
        return true;
    }

    inline bool lessThan(Vector<T, D> other) {
        [ForceUnroll]
        for (int d = 0; d < D; d++) {
            if (data[d] >= other.data[d]) return false;
        }
        return true;
    }

    inline bool lessThanOrEquals(Vector<T, D> other) {
        [ForceUnroll]
        for (int d = 0; d < D; d++) {
            if (data[d] > other.data[d]) return false;
        }
        return true;
    }

    inline Vector<T, D> sqrt() {
        T[D] out;
        [ForceUnroll]
        for (int d = 0; d < D; d++) {
            out[d] = sqrt(data[d]);
        }

        return Vector<T, D>(out);
    }

    inline Vector<T, D> lerp(Vector<T, D> other, T t) {
        T[D] out;
        [ForceUnroll]
        for (int d = 0; d < D; d++) {
            out[d] = data[d] * t + other.data[d] * (T(1) - t);
        }
        return Vector<T, D>(out);
    }
}

inline Vector<T, D> operator+<T:Float, let D:int>(Vector<T, D> a, Vector<T, D> b) {
    return a.add(b);
}

inline Vector<T, D> operator+<T:Float, let D:int>(T a, Vector<T, D> b) {
    return b.add(a);
}



inline Vector<T, D> operator+<T : Float, let D : int>(Vector<T, D> a, T b) {
    return a.add(b);
}

inline Vector<T, D> operator*<T : Float, let D : int>(Vector<T, D> a, Vector<T, D> b) {
    return a.mul(b);
}

inline Vector<T, D> operator*<T : Float, let D : int>(Vector<T, D> a, T b) {
    return a.mul(b);
}

inline Vector<T, D> operator*<T : Float, let D : int>(T a, Vector<T, D> b) {
    return b.mul(a);
}


inline Vector<T, D> operator/<T : Float, let D : int>(Vector<T, D> a, T b) {
    return a.div(b);
}



inline Vector<T, D> operator/<T : Float, let D : int>(T a, Vector<T, D> b) {
    return b.div(a);
}

inline Vector<T, D> operator-<T : Float, let D : int>(Vector<T, D> a, T b) {
    return a.sub(b);
}

inline Vector<T, D> operator-<T : Float, let D : int>(T a, Vector<T, D> b) {
    return b.sub(a);
}

inline Vector<T, D> operator%<T : Float, let D : int>(Vector<T, D> a, T b) {
    return a.mod(b);
}

inline Vector<T, D> operator%<T : Float, let D : int>(T a, Vector<T, D> b) {
    return b.mod(a);
}

inline Vector<T, D> operator-<T : Float, let D : int>(Vector<T, D> a) {
    return a.neg();
}

inline Vector<T, D> load_vector<T : Float, let D : int>(RWTensor<T, 2> data, int index) {
    
    T[D] out;
    [ForceUnroll]
    for(int d=0; d<D; d++) {
        out[d] = data[index, d];
    }
    return Vector<T, D>(out);
}


inline void store_vector<T : Float, let D : int>(RWTensor<T, 2> data, Vector<T, D> v, int index) {
    [ForceUnroll]
    for (int d = 0; d < D; d++) {
        data.set({index, d}, v.data[d]);
    }
}
