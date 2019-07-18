package com.xixici;

import java.util.Stack;

/**
 * Created by xixici
 * Date: 2019/3/13
 * Project Name: sword-offer-scala-sbt
 * Project URL: https://github.com/xixici/sword-offer-scala
 */
public class Q20 {

    public boolean IsPopOrder(int[] pushA, int[] popA) {
        if (pushA.length != 0 && popA.length != 0) {
            Stack<Integer> stack = new Stack();
            int j = 0;

            for (int i = 0; i < popA.length; ++i) {
                stack.push(pushA[i]);

                while (j < popA.length && (Integer) stack.peek() == popA[j]) {
                    stack.pop();
                    ++j;
                }
            }

            return stack.empty();
        } else {
            return false;
        }
    }
}
