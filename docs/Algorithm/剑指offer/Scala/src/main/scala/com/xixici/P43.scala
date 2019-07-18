package com.xixici

/**
  * Created by xixici
  * Date: 2019/3/30 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
object P43 {
  def LeftRotateString(str: String, n: Int) = {
    val chars: Array[Char] = str.toCharArray
    reverse(chars, 0, n - 1)
    reverse(chars, n, chars.length - 1)
    reverse(chars, 0, chars.length - 1)
    chars.mkString("")
  }

  def reverse(chars: Array[Char], start: Int, end: Int): Unit = {
    var i = start
    var j = end
    while (i < j) {
      val tmp = chars(i)
      chars(i) = chars(j)
      chars(j) = tmp
      i += 1
      j -= 1
    }

  }
}
