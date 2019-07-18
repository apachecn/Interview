package com.xixici

/**
  * Created by xixici
  * Date: 2019/3/13 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
object P08 {
  def Jump(n: Int): Int = {
    if (n == 1) {
      return 1
    }
    if (n == 2) {
      return 2
    }

    Jump(n - 1) + Jump(n - 2)
  }
}
