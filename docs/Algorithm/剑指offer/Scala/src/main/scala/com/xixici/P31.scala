package com.xixici

/**
  * Created by xixici
  * Date: 2019/3/22 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
object P31 {

  def NumberOf1Between1AndN_Solution(n: Int) = {
    var count = 0
    var i = 1
    while (i < n) {
      count += (n / (i * 10)) * i + Math.min(Math.max(n % (i * 10) - i + 1, 0), i)
      i *= 10
    }
    count
  }
}
