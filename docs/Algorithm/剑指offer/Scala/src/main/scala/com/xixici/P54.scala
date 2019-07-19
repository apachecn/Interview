package com.xixici

/**
  * Created by xixici
  * Date: 2019/4/29 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
object P54 {

  val hashtable = new Array[Int](256)
  val s = new StringBuffer()

  //Insert one char from stringstream
  def Insert(ch: Char): Unit = {
    s.append(ch)
    if (hashtable(ch) == 0)
      hashtable(ch) = 1
    else hashtable(ch) += 1
  }

  //return the first appearence once char in current stringstream
  def FirstAppearingOnce(): Char = {
    val str = s.toString.toCharArray
    for (c <- str) {
      if (hashtable(c) == 1)
        return c
    }
    '#'
  }
}
