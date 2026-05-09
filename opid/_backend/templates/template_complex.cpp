#include "../src/extra.h"

extern "C" {
void model_real(double* a, double* b, double* c, long n) {
    Vector u(n, true, a);
    Vector v(n, true, b);
    {{ model_real_code }}
    memcpy(c, model_real_result._data, n * sizeof(scalar_t));
}

void model_complex(double* a, double* b, double* c, long n) {
    Vector u(n, true, a);
    Vector v(n, true, b);
    {{ model_complex_code }}
    memcpy(c, model_complex_result._data, n * sizeof(scalar_t));
}
}
