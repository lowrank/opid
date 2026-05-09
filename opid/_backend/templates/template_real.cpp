#include "../src/extra.h"

extern "C" {
void model(double* a, double* b, long n) {
    Vector u(n, true, a);
    {{ model_code }}
    memcpy(b, model_result._data, n * sizeof(scalar_t));
}
}
