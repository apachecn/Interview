package com.xixici

import scala.collection.mutable

/**
  * Created by xixici
  * Date: 2019/3/30 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
object P41 {
  def FindContinuousSequence(sum: Int) = {
    var result = mutable.Buffer[Int]()
    var plow = 1
    var phigh = 2
    while (phigh > plow) {
      val cur = (phigh + plow) * (phigh - plow + 1) / 2
      if (cur == sum) {
        val list = mutable.Buffer[Int]()
        for (i <- plow to phigh) {
          list += i
        }
        result = result ++ list
        plow += 1
      } else if (cur < sum) {
        phigh += 1
      } else {
        plow += 1
      }

    }
    result.toList
  }
}
