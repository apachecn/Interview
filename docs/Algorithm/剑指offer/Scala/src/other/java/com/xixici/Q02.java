package com.xixici;

public class Q02 {
    public String replaceSpace(StringBuffer str) {
        StringBuffer newStr = new StringBuffer();
        for (int i = 0; i < str.length(); i++) {
            if (str.charAt(i) == ' ') {
                newStr.append("%20");
            } else {
                newStr.append(str.charAt(i));
            }
        }
        return newStr.toString();
    }
}