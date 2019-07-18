package com.xixici

/**
  * Created by xixici
  * Date: 2019/4/26 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
object P44 {
  def ReverseSentence(str: String) = {
    val chars = str.toCharArray
    val charsLen = chars.length
    reverse(chars, 0, charsLen - 1)
    var blank = -1
    for (i <- 0 until charsLen) {
      if (chars(i) == ' ') {
        val blankNext = i
        reverse(chars, blank + 1, blankNext - 1)
        blank = blankNext

      }
    }
    reverse(chars, blank + 1, charsLen - 1)
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