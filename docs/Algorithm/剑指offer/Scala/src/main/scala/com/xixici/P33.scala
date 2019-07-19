package com.xixici

/**
  * Created by xixici
  * Date: 2019/3/21 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
object P33 {
  def GetUglyNumber_Solution(index: Int) = {
    val list = new collection.mutable.ListBuffer[Int]()
    list.append(1)
    var t2, t3, t5 = 0
    for (i <- 0 until index) {
      list.append(math.min(math.min(list(t2) * 2, list(t3) * 3), list(t5) * 5))
      if (list(i + 1) == list(t2) * 2) t2 += 1
      if (list(i + 1) == list(t3) * 3) t3 += 1
      if (list(i + 1) == list(t5) * 5) t5 += 1
    }
    list(index - 1)
  }
}
