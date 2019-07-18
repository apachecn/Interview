package com.xixici

/**
  * Created by xixici
  * Date: 2019/3/20 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
object P29 {
  def GetLeastNumbers_Solution(input: Array[Int], k: Int): List[Int] = {
    val list = new collection.mutable.ListBuffer[Int]()
    if (input == null || input.length <= 0 || input.length < k) {
      return list.toList
    }
    // normal sort
    for (i <- 0 until k) {
      for (j <- 0 until input.length - i - 1) {
        if (input(j) < input(j + 1)) {
          val temp = input(j)
          input(j) = input(j + 1)
          input(j + 1) = temp
        }
      }
      list.append(input(input.length - i - 1))
    }
    list.toList
  }
}
