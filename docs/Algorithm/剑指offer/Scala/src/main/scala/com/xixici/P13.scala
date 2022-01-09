package com.xixici

/**
  * Created by xixici
  * Date: 2019/3/13 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
object P13 {
  def reOrderArray(array: Array[Int]) = {
    for (i <- 0 until array.length) {
      if (array(i) % 2 == 1) {
        val tmp = array(i)
        var j = i - 1
        while (j >= 0 && array(j) % 2 == 0) {
          array(j + 1) = array(j)
          j -= 1
        }
        array(j + 1) = tmp
      }
    }
    array
  }
}
