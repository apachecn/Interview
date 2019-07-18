package com.xixici

/**
  * Created by xixici
  * Date: 2019/3/15 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
object P23 {
  def VerifySquenceOfBST(sequence: Array[Int]): Boolean = {

    if (sequence.length == 0) {
      return false;
    }
    if (sequence.length == 1) {
      return true;
    }
    return judge(sequence, 0, sequence.length - 1);
  }

  def judge(a: Array[Int], start: Int, end: Int): Boolean = {
    if (start >= end) {
      return true;
    }
    var i = start;
    while (a(i) < a(end)) {
      i += 1
    }
    for (j <- i until end) {
      if (a(j) < a(end)) {
        return false;
      }
    }
    return judge(a, start, i - 1) && judge(a, i, end - 1)
  }
}
