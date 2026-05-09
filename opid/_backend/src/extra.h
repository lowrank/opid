#ifndef DERIVATIVE_H
#define DERIVATIVE_H

#include "linalg.h"
#include <fftw3.h>

Vector operator+(const Vector& lhs, const Vector& rhs) {
    assert(lhs._row == rhs._row);
    Vector res(lhs._row);

    for (int i = 0; i < lhs._row; i++) {
        res(i) = lhs(i) + rhs(i);
    }
    return res;
}

Vector operator-(const Vector& lhs, const Vector& rhs) {
    assert(lhs._row == rhs._row);
    Vector res(lhs._row);

    for (int i = 0; i < lhs._row; i++) {
        res(i) = lhs(i) - rhs(i);
    }
    return res;
}


Vector operator-(const Vector& lhs) {
    Vector res(lhs._row);
    for (int i = 0; i < lhs._row; i++) {
        res(i) = -lhs(i);
    }
    return res;
}

Vector operator*(const Vector& lhs, const Vector& rhs){
    assert(lhs._row == rhs._row);
    Vector res(lhs._row);

    for (int i = 0; i < lhs._row; i++) {
        res(i) = lhs(i) * rhs(i);
    }
    return res;
}

Vector operator*(const scalar_t a, const Vector& rhs){
    Vector res(rhs._row);
    for (int i = 0; i < rhs._row; i++) {
        res(i) = a * rhs(i);
    }
    return res;
    return res;
}

Vector operator*(const Vector& lhs, const scalar_t a) {
    Vector res(lhs._row);
    for (int i = 0; i < lhs._row; i++) {
        res(i) = lhs(i) * a;
    }
    return res;
}

Vector sin(const Vector& x) {
    Vector res(x._row);
    for (int i = 0; i < x._row; i++) {
        res(i) = sin(x(i));
    }
    return res;
}

Vector cos(const Vector& x) {
    Vector res(x._row);
    for (int i = 0; i < x._row; i++) {
        res(i) = cos(x(i));
    }
    return res;
}

Vector tan(const Vector& x) {
    Vector res(x._row);
    for (int i = 0; i < x._row; i++) {
        res(i) = tan(x(i));
    }
    return res;
}

Vector log(const Vector& x) {
    Vector res(x._row);
    for (int i = 0; i < x._row; i++) {
        res(i) = log(x(i));
    }
    return res;
}

Vector exp(const Vector& x) {
    Vector res(x._row);
    for (int i = 0; i < x._row; i++) {
        res(i) = exp(x(i));
    }
    return res;
}

Vector atan(const Vector& x) {
    Vector res(x._row);
    for (int i = 0; i < x._row; i++) {
        res(i) = atan(x(i));
    }
    return res;
}

Vector asin(const Vector& x) {
    Vector res(x._row);
    for (int i = 0; i < x._row; i++) {
        res(i) = asin(x(i));
    }
    return res;
}

Vector acos(const Vector& x) {
    Vector res(x._row);
    for (int i = 0; i < x._row; i++) {
        res(i) = acos(x(i));
    }
    return res;
}

Vector pow(const Vector& x, scalar_t n) {
    Vector res(x._row);
    for (int i = 0; i < x._row; i++) {
        res(i) = pow(x(i), n);
    }
    return res;
}

Vector fabs(const Vector& x) {
    Vector res(x._row);
    for (int i = 0; i < x._row; i++) {
        res(i) = fabs(x(i));
    }
    return res;
}

Vector D(const Vector& x) {
    Vector res(x._row);
    Vector out(x._row);
    fftw_plan p_forward, p_backward;

    p_forward = fftw_plan_r2r_1d(x._row, x._data, out._data, FFTW_R2HC, FFTW_ESTIMATE);
    p_backward = fftw_plan_r2r_1d(x._row, out._data, res._data, FFTW_HC2R, FFTW_ESTIMATE);

    fftw_execute(p_forward);

    out(0) = 0;
    for (int i = 1; i < x._row / 2; ++i) {
        scalar_t tmp = out(i);
        out(i) = -out(x._row - i) * i;
        out(x._row - i) = tmp * i;
    }

    fftw_execute(p_backward);

    fftw_destroy_plan(p_backward);
    fftw_destroy_plan(p_forward);

    res = res * (1.0/x._row);

    return res;
}


#endif // DERIVATIVE_H
