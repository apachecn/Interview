package com.xixici

/**
  * Created by xixici
  * Date: 2019/3/5 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
object P01 {
  def find(target: Int, array: Array[Array[Int]]) = {
    val arrayLen = array.length
    var res = false
    for (i: Int <- 0 until arrayLen) {
      for (j: Int <- 0 until array(i).length) {
        if (target == array(i)(j)) {
          res = true
        }
      }
    }
    res
  }
}
