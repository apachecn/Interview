package com.xixici;

public class Q08 {
    public int JumpFloor(int target) {
        int a = 1, b = 1, v = 0;
        if (target == 1) {
            v = 1;
        }
        if (target == 2) {
            v = 2;
        }
        if (target >= 3) {
            for (int i = 3; i <= target; i++) {
                v = a + b;
                a = b;
                b = v;
            }
        }
        return v;
    }

}
