package com.xixici

/**
  * Created by xixici
  * Date: 2019/3/21 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
object P32 {
  def PrintMinNumber(numbers: Array[Int]): String = {
    var str: String = ""
    for (i <- numbers.indices) {
      for (j <- i + 1 until numbers.length) {
        val a = (numbers(i) + "" + numbers(j)).toInt
        val b = (numbers(j) + "" + numbers(i)).toInt
        if (a > b) {
          val t = numbers(i)
          numbers(i) = numbers(j)
          numbers(j) = t
        }
      }
    }
    for (i <- numbers.indices) {
      str += numbers(i).toString
    }
    str
  }

}
