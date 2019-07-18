package com.xixici

/**
  * Created by xixici
  * Date: 2019/3/11 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
object P07 {
  def Fibonacci(n: Int): Int = {
    var res = 0
    if (n == 1) {
      return 1
    }
    if (n == 0) {
      return 0
    }

    Fibonacci(n - 1) + Fibonacci(n - 2)
  }
}
