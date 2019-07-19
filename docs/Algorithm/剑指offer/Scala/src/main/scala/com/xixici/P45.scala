package com.xixici

/**
  * Created by xixici
  * Date: 2019/4/26 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
object P45 {
  def isContinuous(numbers: Array[Int]): Boolean = {
    val numLen = numbers.length
    if (numLen != 5) return false
    var min = 14
    var max = -1
    var flag = 0
    for (i <- 0 until numLen if numbers(i) != 0) {
      val number = numbers(i)
      if (number < 0 || number > 13) return false
      //      if (number == 0) return
      if (((flag >> number) & 1) == 1) return false
      flag |= (1 << number)
      if (number > max) max = number
      if (number < min) min = number
      if (max - min >= 5) return false
    }
    true

  }
}
