package com.xixici

import scala.collection.mutable.ListBuffer

/**
  * Created by xixici
  * Date: 2019/4/26 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
object P46 {
  def LastRemaining_Solution(n: Int, m: Int) = {
    val list = ListBuffer[Int]()
    for (i <- 0 until n) {
      list.append(i)
    }

    var bt = 0
    while (list.size > 1) {
      bt = (bt + m - 1) % list.size
      list.remove(bt)
    }

    if (list.size == 1)
      list(0)
    else {
      -1
    }
  }
}
