package com.xixici;

import java.util.ArrayList;

/**
 * Created by xixici
 * Date: 2019/3/13
 * Project Name: sword-offer-scala-sbt
 * Project URL: https://github.com/xixici/sword-offer-scala
 */

public class Q19 {

    public ArrayList<Integer> printMatrix(int[][] matrix) {
        int row = matrix.length;
        int col = matrix[0].length;
        if (row != 0 && col != 0) {
            ArrayList<Integer> list = new ArrayList();
            int left = 0;
            int top = 0;
            int bottom = row - 1;

            for (int right = col - 1; left <= right && top <= bottom; --bottom) {
                int k;
                for (k = left; k <= right; ++k) {
                    list.add(matrix[top][k]);
                }

                for (k = top + 1; k <= bottom; ++k) {
                    list.add(matrix[k][right]);
                }

                if (top != bottom) {
                    for (k = right - 1; k >= left; --k) {
                        list.add(matrix[bottom][k]);
                    }
                }

                if (left != right) {
                    for (k = bottom - 1; k > top; --k) {
                        list.add(matrix[k][left]);
                    }
                }

                ++top;
                ++left;
                --right;
            }

            return list;
        } else {
            return null;
        }
    }
}


