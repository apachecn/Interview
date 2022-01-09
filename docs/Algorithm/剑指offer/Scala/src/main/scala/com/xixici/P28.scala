package com.xixici

import scala.collection.mutable

/**
  * Created by xixici
  * Date: 2019/3/18 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
object P28 {
  def MoreThanHalfNum_Solution(arr: Array[Int]): Int = {
    val hashMap = new mutable.HashMap[Int, Int]()
    val len = arr.length
    for (i <- 0 until len) {
      val tmp = hashMap.getOrElseUpdate(arr(i), 0)
      if (tmp == 0) {
        hashMap.put(arr(i), 1)
      } else {
        hashMap.put(arr(i), tmp + 1)
      }
      if (tmp + 1 > len / 2) return arr(i)

    }
    0
  }
}
