package com.xixici

/**
  * Created by xixici
  * Date: 2019/3/13 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
object P12 {
  def Power(base: Double, exponent: Int): Double = {
    var res = base
    var exponentNew = exponent
    if (exponent == 0) {
      res = 0
    }
    if (exponent < 0) {
      for (i <- 1 until exponent) {
        res *= base
      }
      res = 1 / res
    }
    if (exponent > 0) {
      for (i <- 1 until exponent) {
        res *= base
      }
    }
    res
  }
}
