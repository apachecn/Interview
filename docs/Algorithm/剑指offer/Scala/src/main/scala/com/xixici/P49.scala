package com.xixici

/**
  * Created by xixici
  * Date: 2019/3/16 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
object P49 {
  def StrToInt(str: String): Int = {
    if (str.length == 0) {
      return 0
    }
    var sum = 0
    var label = 1
    if (str.charAt(0) == '+') label = 1
    if (str.charAt(0) == '-') label = -1

    for (s <- str) {
      if (s > '9' || s < '0') {
        sum = 0
      }
      else {
        sum = sum * 10 + s - '0'
      }
    }
    label * sum
  }
}
