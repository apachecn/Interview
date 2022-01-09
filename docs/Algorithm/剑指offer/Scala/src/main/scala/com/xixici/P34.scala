package com.xixici

/**
  * Created by xixici
  * Date: 2019/3/22 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
object P34 {
  def FirstNotRepeatingChar(str: String): Int = {
    val chars = str.toCharArray
    val map = new Array[Int](256)
    for (i <- 0 until str.length) {
      map(chars(i)) += 1
    }
    for (i <- 0 until chars.length) {
      if (map(chars(i)) == 1) return i
    }
    -1
  }
}
