package com.xixici

/**
  * Created by xixici
  * Date: 2019/3/18 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
object P50 {
  def duplicate(numbers: Array[Int], length: Int, duplication: Array[Int]): Boolean = {

    val k = new Array[Boolean](length)
    for (i <- 0 until length) {
      if (k(numbers(i))) {
        duplication(0) = numbers(i)
        println(duplication(0))
        return true
      }
      k(numbers(i)) = true
    }
    false
  }
}
