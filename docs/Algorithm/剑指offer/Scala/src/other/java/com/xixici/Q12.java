package com.xixici;

public class Q12 {
    public double Power(double base, int exponent) {
        int n = exponent;
        double result = base;
        if (exponent < 0) {
            n = -exponent;
        }
        if (n == 0) {
            result = 1;
        }
        for (int i = 1; i < n; i++) {
            result *= base;
        }
        return exponent < 0 ? 1 / result : result;
    }
}
