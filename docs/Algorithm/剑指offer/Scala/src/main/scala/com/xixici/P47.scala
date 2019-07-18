package com.xixici

/**
  * Created by xixici
  * Date: 2019/3/16 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
object P47 {
  def Sum_Solution(n: Int): Int = {
    //    (1 + n) * n / 2
    var sum = n
    (sum > 0) && {
      sum += Sum_Solution(n - 1)
      sum > 0
    }
    sum
  }
}
